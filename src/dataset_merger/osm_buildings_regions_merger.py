from typing import Any, Dict, List
from collections import deque
import math

from tqdm import tqdm
import pandas as pd
import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape
from shapely.ops import unary_union

from .dataset_merger import DatasetMerger
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter
from ..dataset_merger import GeoPandasIdDatasetJoiner
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

def make_valid(geometry):
    return shapely.make_valid(geometry, method='structure', keep_collapsed=False)

class OSMBuildingsRegionsMerger(DatasetMerger):
    def __init__(self, *, buildings_dataset_key: str,
                        regions_dataset_key: str,
                        buildings_col: str,
                        buildings_id_col: str,
                        buildings_features_col: str,
                        regions_col: str,
                        regions_id_col: str,
                        site_contour_col: str,
                        massing_col: str,
                        building_properties_col: str,
                        base_point_col: str,
                        max_region_buildings: int,
                        min_building_footprint_ratio: float,
                        user_eps: float,
                        ops_eps: float) -> None:
        self.buildings_dataset_key = buildings_dataset_key
        self.regions_dataset_key = regions_dataset_key
        self.buildings_col = buildings_col
        self.buildings_id_col = buildings_id_col
        self.buildings_features_col = buildings_features_col
        self.regions_col = regions_col
        self.regions_id_col = regions_id_col
        self.site_contour_col = site_contour_col
        self.massing_col = massing_col
        self.building_properties_col = building_properties_col
        self.base_point_col = base_point_col
        self.max_region_buildings = max_region_buildings
        self.min_building_footprint_ratio = min_building_footprint_ratio
        self.user_eps = user_eps
        self.ops_eps = ops_eps

        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()
        
    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        buildings_dataset = datasets[self.buildings_dataset_key]
        regions_dataset = datasets[self.regions_dataset_key]

        tqdm.pandas()
        #----------GET REGIONS-BUILDINGS SPATIAL GRAPH----------
        print("----------GET REGIONS-BUILDINGS SPATIAL GRAPH----------")
        regions = pd.DataFrame(regions_dataset)
        print(len(regions))
        def get_contour(row):
            polygons = [[(p[0], p[1]) for p in polygon] for polygon in row[self.regions_col]]
            contour = self.polygons_to_shapely_converter(polygons=polygons)
            return contour
        regions[self.regions_col] = regions.progress_apply(lambda row: get_contour(row), axis=1)

        buildings = pd.DataFrame(buildings_dataset)
        print(len(buildings))
        def get_footprint(row):
            shapely_polygons = []
            for e in row[self.buildings_col]["massing"]:
                polygons = [[(p[0], p[1]) for p in polygon] for polygon in e["polygons"]]
                shapely_polygons.append(self.polygons_to_shapely_converter(polygons=polygons))
            footprint = unary_union(shapely_polygons)
            return footprint
        buildings["footprint"] = buildings.progress_apply(lambda row: get_footprint(row), axis=1)
        buildings_dict = {row[self.buildings_id_col]:row.to_dict() for _, row in buildings.iterrows()}
        
        id_joiner = GeoPandasIdDatasetJoiner(
            join_dataset_key="join_dataset",
            join_dataset_geo_col="footprint",
            join_dataset_id_col=self.buildings_id_col,
            main_dataset_key="main_dataset",
            main_dataset_geo_col=self.regions_col,
            predicate="intersects",
            output_col="building_ids"
        )
        regions = id_joiner(datasets={"main_dataset":regions.to_dict("list"), "join_dataset":buildings.to_dict("list")})
        regions = pd.DataFrame(regions)

        #----------MERGE REGIONS----------
        print("----------MERGE REGIONS----------")

        def get_buildings_to_regions(regions):
            building_to_regions = {}
            for _, row in tqdm(regions.iterrows(), total=len(regions)):
                for building_id in row["building_ids"]:
                    if building_id not in building_to_regions:
                        building_to_regions[building_id] = []
                    building_to_regions[building_id].append(row[self.regions_id_col])
            return building_to_regions

        def get_regions_graph(building_to_regions):
            regions_graph = {}
            for region_ids in building_to_regions.values():
                for region_id in region_ids:
                    if region_id not in regions_graph:
                        regions_graph[region_id] = set()
                    regions_graph[region_id].update([i for i in region_ids if i != region_id])
            return regions_graph

        def get_node_to_component(regions_graph):
            node_to_component = {}
            visited = set()
            for region_id in tqdm(list(regions_graph.keys())):
                if region_id in visited:
                    continue

                component = []
                queue = deque([region_id])
                visited.add(region_id)
                while len(queue) > 0:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in regions_graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                component_id = component[0]
                for node in component:
                    node_to_component[node] = component_id
            return node_to_component

        def get_regions_dict(regions, node_to_component):
            regions_dict = {}
            for _, row in tqdm(regions.iterrows(), total=len(regions)):
                if row[self.regions_id_col] in node_to_component:
                    component = node_to_component[row[self.regions_id_col]]
                    if component in regions_dict:
                        regions_dict[component][self.regions_col] = regions_dict[component][self.regions_col].union(row[self.regions_col])
                        regions_dict[component]["building_ids"] = list(set(regions_dict[component]["building_ids"]).union(set(row["building_ids"])))
                    else:
                        regions_dict[component] = row.to_dict()
                else:
                    regions_dict[row[self.regions_id_col]] = row.to_dict()
            return regions_dict

        def merge_shared_buildings_regions(regions_dict):
            region_cols = [self.regions_id_col, self.regions_col, "building_ids"]
            regions_dataset = {k:[] for k in region_cols}
            for key, value in tqdm(regions_dict.items(), total=len(list(regions_dict.keys()))):
                for k in region_cols:
                    if k == self.regions_id_col:
                        regions_dataset[k].append(key)
                    else:
                        regions_dataset[k].append(value[k])
            regions = pd.DataFrame(regions_dataset)
            return regions

        buildings_to_regions = get_buildings_to_regions(regions)
        regions_graph = get_regions_graph(buildings_to_regions)
        node_to_component = get_node_to_component(regions_graph)
        regions_dict = get_regions_dict(regions, node_to_component)
        regions = merge_shared_buildings_regions(regions_dict)

        def filter_buildings(row):
            if len(row["building_ids"]) == 0:
                return row["building_ids"]
            ids = row["building_ids"]
            new_ids = []
            for i in ids:
                footprint = buildings_dict[i]["footprint"]
                if footprint.within(row[self.regions_col]):
                    new_ids.append(i)
            return new_ids
        regions["building_ids"] = regions.progress_apply(lambda row: filter_buildings(row), axis=1)

        def get_region_outer_contour(region):
            if type(region) == Polygon:
                return Polygon(region.exterior)
            if type(region) == MultiPolygon:
                return unary_union([Polygon(g.exterior) for g in region.geoms])
            return region

        def get_region_building_footprint_area(region):
            footprint_area = 0
            for building_id in region["building_ids"]:
                if building_id not in buildings_dict:
                    continue
                footprint_area += buildings_dict[building_id]["footprint"].area
            return footprint_area

        def get_region_stats(regions):
            stats = {}
            for _, row in tqdm(regions.iterrows(), total=len(regions)):
                region_area = row[self.regions_col].area
                footprint_area = get_region_building_footprint_area(row)
                stats[row[self.regions_id_col]] = {
                    "area": region_area,
                    "n_buildings": len(set(row["building_ids"])),
                    "footprint_area": footprint_area,
                    "footprint_ratio": footprint_area / region_area if region_area > 0 else float("inf"),
                }
            return stats

        def get_inner_regions(regions):
            outer_regions = pd.DataFrame(regions)
            outer_regions[f"{self.regions_col}_outer"] = outer_regions.progress_apply(lambda row: get_region_outer_contour(row[self.regions_col]), axis=1)

            id_joiner = GeoPandasIdDatasetJoiner(
                join_dataset_key="join_dataset",
                join_dataset_geo_col=f"{self.regions_col}_outer",
                join_dataset_id_col=self.regions_id_col,
                main_dataset_key="main_dataset",
                main_dataset_geo_col=f"{self.regions_col}_outer",
                predicate="within",
                output_col="inner_region_ids"
            )
            outer_regions = id_joiner(datasets={"main_dataset":outer_regions.to_dict("list"), "join_dataset":outer_regions.to_dict("list")})
            outer_regions = pd.DataFrame(outer_regions)
            outer_regions["inner_region_ids"] = outer_regions.progress_apply(
                lambda row: [i for i in row["inner_region_ids"] if i != row[self.regions_id_col]],
                axis=1
            )
            return outer_regions[[self.regions_id_col, "inner_region_ids"]]

        def get_region_parents(inner_regions):
            region_parents = {}
            for _, row in tqdm(inner_regions.iterrows(), total=len(inner_regions)):
                parent_id = row[self.regions_id_col]
                for child_id in row["inner_region_ids"]:
                    if child_id not in region_parents:
                        region_parents[child_id] = []
                    region_parents[child_id].append(parent_id)
            return region_parents

        def get_region_tree(region_parents, region_stats):
            parent_by_region = {}
            children_by_region = {}
            for child_id, parent_ids in region_parents.items():
                if len(parent_ids) == 0:
                    continue
                parent_id = min(parent_ids, key=lambda i: region_stats[i]["area"])
                parent_by_region[child_id] = parent_id
                if parent_id not in children_by_region:
                    children_by_region[parent_id] = []
                children_by_region[parent_id].append(child_id)
            return parent_by_region, children_by_region

        def get_subtree_stats(regions, children_by_region):
            region_by_id = {row[self.regions_id_col]:row.to_dict() for _, row in regions.iterrows()}
            subtree_stats = {}

            def get_stats(region_id):
                if region_id in subtree_stats:
                    return subtree_stats[region_id]

                row = region_by_id[region_id]
                region_ids = [region_id]
                building_ids = list(row["building_ids"])
                geometry = row[self.regions_col]

                for child_id in children_by_region.get(region_id, []):
                    child_stats = get_stats(child_id)
                    region_ids.extend(child_stats["region_ids"])
                    building_ids = list(set(building_ids).union(set(child_stats["building_ids"])))
                    geometry = geometry.union(child_stats["geometry"])

                footprint_area = 0
                for building_id in building_ids:
                    if building_id in buildings_dict:
                        footprint_area += buildings_dict[building_id]["footprint"].area

                subtree_stats[region_id] = {
                    "region_ids": region_ids,
                    "building_ids": building_ids,
                    "n_buildings": len(building_ids),
                    "geometry": geometry,
                    "footprint_ratio": footprint_area / geometry.area,
                }
                return subtree_stats[region_id]

            for region_id in tqdm(list(region_by_id.keys())):
                get_stats(region_id)
            return subtree_stats

        def get_valid_roots_by_building_count(root_ids, children_by_region, subtree_stats):
            valid_roots = []
            trash_region_ids = set()

            def cut(region_id):
                if subtree_stats[region_id]["n_buildings"] <= self.max_region_buildings:
                    valid_roots.append(region_id)
                    return
                trash_region_ids.add(region_id)
                for child_id in children_by_region.get(region_id, []):
                    cut(child_id)

            for root_id in tqdm(root_ids):
                cut(root_id)
            return valid_roots, trash_region_ids

        def get_valid_roots_by_footprint_ratio(root_ids, children_by_region, subtree_stats):
            valid_roots = []
            trash_region_ids = set()

            def cut(region_id):
                if subtree_stats[region_id]["footprint_ratio"] >= self.min_building_footprint_ratio:
                    valid_roots.append(region_id)
                    return
                trash_region_ids.add(region_id)
                for child_id in children_by_region.get(region_id, []):
                    cut(child_id)

            for root_id in tqdm(root_ids):
                cut(root_id)
            return valid_roots, trash_region_ids

        def get_inner_regions_dict(valid_roots, subtree_stats):
            regions_dict = {}
            for root_id in tqdm(valid_roots):
                regions_dict[root_id] = {
                    self.regions_id_col: root_id,
                    self.regions_col: subtree_stats[root_id]["geometry"],
                    "building_ids": subtree_stats[root_id]["building_ids"],
                }
            return regions_dict

        def add_empty_regions(regions_dict, empty_region_ids, regions):
            regions_by_id = {row[self.regions_id_col]: row.to_dict() for _, row in regions.iterrows()}
            for region_id in tqdm(empty_region_ids):
                if region_id in regions_dict or region_id not in regions_by_id:
                    continue
                regions_dict[region_id] = {
                    self.regions_id_col: region_id,
                    self.regions_col: regions_by_id[region_id][self.regions_col],
                    "building_ids": [],
                }
            return regions_dict

        def merge_inner_regions(regions_dict):
            region_cols = [self.regions_id_col, self.regions_col, "building_ids"]
            regions_dataset = {k:[] for k in region_cols}
            for key, value in tqdm(regions_dict.items(), total=len(list(regions_dict.keys()))):
                for k in region_cols:
                    if k == self.regions_id_col:
                        regions_dataset[k].append(key)
                    else:
                        regions_dataset[k].append(value[k])
            regions = pd.DataFrame(regions_dataset)
            return regions

        inner_regions = get_inner_regions(regions)
        region_parents = get_region_parents(inner_regions)
        region_stats = get_region_stats(regions)
        parent_by_region, children_by_region = get_region_tree(region_parents, region_stats)
        root_ids = [region_id for region_id in regions[self.regions_id_col].values.tolist() if region_id not in parent_by_region]
        subtree_stats = get_subtree_stats(regions, children_by_region)

        building_count_roots, building_count_trash = get_valid_roots_by_building_count(root_ids, children_by_region, subtree_stats)
        footprint_ratio_roots, footprint_ratio_trash = get_valid_roots_by_footprint_ratio(building_count_roots, children_by_region, subtree_stats)
        inner_regions_dict = get_inner_regions_dict(
            footprint_ratio_roots,
            subtree_stats
        )
        """inner_regions_dict = add_empty_regions(
            inner_regions_dict,
            set(footprint_ratio_trash),
            regions
        )"""
        regions = merge_inner_regions(inner_regions_dict)

        def get_circularity(polygon):
            return 4 * math.pi * polygon.area / (polygon.length ** 2) if polygon.length > 0 else 0
        def get_compactness(polygon):
            return polygon.area / polygon.convex_hull.area if polygon.convex_hull.area > 0 else 0
        def get_inscribed_circle_ratio(polygon):
            return shapely.maximum_inscribed_circle(polygon).length
        def is_bad_polygon(polygon):
            if (get_circularity(polygon) < 0.1 and 
                get_inscribed_circle_ratio(polygon) < 2):
                return True
            return False
        regions = regions[regions.progress_apply(lambda row: not is_bad_polygon(row[self.regions_col]), axis=1)]

        #----------TRIM USELESS REGION HOOKS----------
        print("----------TRIM USELESS REGION HOOKS----------")
        HOOK_WIDTH = 5
        FOOTPRINT_SAFE_MARGIN = self.ops_eps

        def collections_to_polygons(geometry):
            polygons = []
            for geom in geometry.geoms:
                geom = make_valid(geom)
                if type(geom) == MultiPolygon:
                    polygons.extend(geom.geoms)
                elif type(geom) == GeometryCollection:
                    polygons.extend(collections_to_polygons(geom))
                elif type(geom) == Polygon:
                    polygons.append(geom)
            return polygons

        def get_polygon_parts(geometry):
            geometry = make_valid(geometry)
            if type(geometry) in [MultiPolygon, GeometryCollection]:
                polygons = collections_to_polygons(geometry)
                if len(polygons) == 0:
                    return []
                return polygons
            elif type(geometry) == Polygon:
                return [geometry]
            else:
                return []

        def get_row_building_footprint(row):
            footprints = [buildings_dict[i]["footprint"] for i in row["building_ids"] if i in buildings_dict]
            if len(footprints) == 0:
                return None
            return unary_union(footprints)

        def trim_region_hooks(row):
            if len(row["building_ids"]) == 0:
                return [row[self.regions_col]]

            region = row[self.regions_col]
            footprint = get_row_building_footprint(row)
            if footprint is None or region is None or region.is_empty:
                return [region]

            try:
                opened = region.buffer(-HOOK_WIDTH).buffer(HOOK_WIDTH)
                opened_parts = get_polygon_parts(opened)
                opened_parts = [
                    polygon
                    for polygon in opened_parts
                    if polygon.area > self.ops_eps and polygon.intersects(footprint)
                ]
                if len(opened_parts) == 0:
                    return [region]

                core = unary_union(opened_parts)
                trimmed = region.intersection(core)
                trimmed = trimmed.union(footprint.buffer(FOOTPRINT_SAFE_MARGIN)).intersection(region)
                trimmed = make_valid(trimmed)
                if trimmed is None or trimmed.is_empty or trimmed.area <= self.ops_eps:
                    return [region]

                uncovered_footprint = footprint.difference(trimmed)
                if not uncovered_footprint.is_empty and uncovered_footprint.area > self.ops_eps:
                    return [region]

                final_parts = get_polygon_parts(trimmed)
                if len(final_parts) == 0:
                    return [region]

                return final_parts
            except:
                return [region]

        regions[self.regions_col] = regions.progress_apply(lambda row: trim_region_hooks(row), axis=1)

        regions_dict = {self.regions_id_col:[], self.regions_col:[]}
        def get_regions(row):
            polygons = row[self.regions_col]
            for polygon in polygons:
                if not polygon.is_valid:
                    try:
                        polygon = make_valid(polygon)
                    except:
                        continue
                try:
                    polygon = Polygon(polygon.exterior.coords)
                except:
                    continue
                regions_dict[self.regions_id_col].append(str(len(regions_dict[self.regions_id_col])))
                regions_dict[self.regions_col].append(polygon)
        regions.progress_apply(lambda row: get_regions(row), axis=1)
        regions = pd.DataFrame(regions_dict)

        def inverse_trim(row):
            region = row[self.regions_col]
            return region.buffer(HOOK_WIDTH).buffer(-HOOK_WIDTH)
        regions[self.regions_col] = regions.progress_apply(lambda row: inverse_trim(row), axis=1)

        def get_exterior(polygon):
            if type(polygon) == Polygon:
                return Polygon(polygon.exterior.coords)
            return None
        regions[self.regions_col] = regions.progress_apply(lambda row: get_exterior(row[self.regions_col]), axis=1)
        regions = regions[regions.progress_apply(lambda row: row[self.regions_col] is not None, axis=1)]

        id_joiner = GeoPandasIdDatasetJoiner(
            join_dataset_key="join_dataset",
            join_dataset_geo_col="footprint",
            join_dataset_id_col=self.buildings_id_col,
            main_dataset_key="main_dataset",
            main_dataset_geo_col=self.regions_col,
            predicate="intersects",
            output_col="building_ids"
        )
        regions = id_joiner(datasets={"main_dataset":regions.to_dict("list"), "join_dataset":buildings.to_dict("list")})
        regions = pd.DataFrame(regions)

        def filter_buildings(row):
            if len(row["building_ids"]) == 0:
                return row["building_ids"]
            ids = row["building_ids"]
            new_ids = []
            for i in ids:
                footprint = buildings_dict[i]["footprint"]
                if footprint.within(row[self.regions_col]):
                    new_ids.append(i)
            return new_ids

        regions["building_ids"] = regions.progress_apply(lambda row: filter_buildings(row), axis=1)

        def filter_regions(row):
            try:
                footprints = [buildings_dict[i]["footprint"] for i in row["building_ids"]]
                footprint = unary_union(footprints)
                region = row[self.regions_col]

                if len(footprints) > self.max_region_buildings:
                    return False
                if footprint.area / region.area < self.min_building_footprint_ratio:
                    return False
                return True
            except:
                return False
        regions = regions[regions.progress_apply(lambda row: filter_regions(row), axis=1)]

        regions = regions[regions.progress_apply(lambda row: get_circularity(row[self.regions_col]) > 0.2, axis=1)]

        def simplify_points(polygon):
            bad_result = None
            eps = self.user_eps
            n_digits = int(abs(math.log10(eps)))

            def simplify_ring(coords):
                new_coords = []
                for x, y in coords:
                    point = (round(x, n_digits), round(y, n_digits))
                    if len(new_coords) == 0:
                        new_coords.append(point)
                        continue
                    elif math.dist(point, new_coords[-1]) > eps:
                        new_coords.append(point)
                if len(new_coords) > 1 and math.dist(new_coords[0], new_coords[-1]) <= eps:
                    new_coords.pop()
                return new_coords

            try:
                exterior = simplify_ring(list(polygon.exterior.coords))
                interiors = [simplify_ring(list(r.coords)) for r in polygon.interiors]
                interiors = [r for r in interiors if len(r) >= 4]
                new_polygon = Polygon(exterior, interiors) if len(interiors) > 0 else Polygon(exterior)
            except:
                return bad_result

            if new_polygon.is_empty or shapely.area(new_polygon) <= eps**2:
                return bad_result

            if not new_polygon.is_valid:
                try:
                    new_polygon = shapely.make_valid(new_polygon, method='structure', keep_collapsed=False)
                except:
                    return bad_result

            return new_polygon
        regions[self.regions_col] = regions.progress_apply(lambda row: simplify_points(row[self.regions_col]), axis=1)
        regions = regions[regions.progress_apply(lambda row: row[self.regions_col] != None, axis=1)]

        #----------GET MASSINGS----------
        print("----------GET MASSINGS----------")

        def get_massing(row):
            massing = [buildings_dict[i][self.buildings_col] for i in row["building_ids"]]
            for i in range(len(massing)):
                massing[i]["id"] = str(i)
            return massing
        regions["massing"] = regions.progress_apply(lambda row: get_massing(row), axis=1)

        def get_building_properties(row):
            building_properties = []
            for i in range(len(row["building_ids"])):
                prop = {"id":str(i)}
                building = buildings_dict[row["building_ids"][i]]
                prop.update(building[self.buildings_features_col])
                building_properties.append(prop)
            return building_properties
        regions[self.building_properties_col] = regions.progress_apply(lambda row: get_building_properties(row), axis=1)

        def is_site_match(row):
            try:
                footprints = [buildings_dict[i]["footprint"] for i in row["building_ids"]]
                footprint = unary_union(footprints)
                site_contour = row[self.regions_col]
                intersection = footprint.intersection(site_contour)
                return abs(intersection.area / footprint.area - 1) < self.ops_eps
            except:
                return False
        regions = regions[regions.progress_apply(lambda row: is_site_match(row), axis=1)]

        regions[self.site_contour_col] = regions.progress_apply(lambda row: self.shapely_to_polygons_converter(polygons=row[self.regions_col]), axis=1)
        regions = regions[regions.progress_apply(lambda row: len(row[self.site_contour_col]) > 0, axis=1)]

        def round_points(row):
            region = row[self.site_contour_col]
            n_digits = int(abs(math.log10(self.user_eps)))
            new_region = []
            for polygon in region:
                new_polygon = [(round(p[0], n_digits), round(p[1], n_digits)) for p in polygon]
                new_region.append(new_polygon)

            massing = row[self.massing_col]
            new_massing = []
            for m in massing:
                new_m = {"id":m["id"], "massing":[]}
                for e in m["massing"]:
                    new_e = {"polygons":[[(round(p[0], n_digits), round(p[1], n_digits)) for p in polygon] for polygon in e["polygons"]], "bottom_elevation":e["bottom_elevation"], "top_elevation":e["top_elevation"]}
                    new_m["massing"].append(new_e)
                new_massing.append(new_m)
            return new_region, new_massing
        regions[[self.site_contour_col, self.massing_col]] = regions.progress_apply(lambda row: round_points(row), axis=1, result_type='expand')
        regions = regions[[self.regions_id_col, self.site_contour_col, self.building_properties_col, self.massing_col]]

        def get_relative_coords(row):
            base_point = row[self.site_contour_col][0][0]
            new_site_contour = [[(p[0]-base_point[0], p[1]-base_point[1]) for p in polygon] for polygon in row[self.site_contour_col]]

            new_massing = []
            for m in row["massing"]:
                new_m = {"id":m["id"], "massing":[]}
                for e in m["massing"]:
                    new_e = {"polygons":[[(p[0]-base_point[0], p[1]-base_point[1]) for p in polygon] for polygon in e["polygons"]], "bottom_elevation":e["bottom_elevation"], "top_elevation":e["top_elevation"]}
                    new_m["massing"].append(new_e)
                new_massing.append(new_m)
            return base_point, new_site_contour, new_massing
        regions[[self.base_point_col, self.site_contour_col, self.massing_col]] = regions.progress_apply(lambda row: get_relative_coords(row), axis=1, result_type='expand')
        regions = regions[[self.regions_id_col, self.base_point_col, self.site_contour_col, self.building_properties_col, self.massing_col]]
        
        def round_points(row):
            region = row[self.site_contour_col]
            n_digits = int(abs(math.log10(self.user_eps)))
            new_region = []
            for polygon in region:
                new_polygon = [(round(p[0], n_digits), round(p[1], n_digits)) for p in polygon]
                new_region.append(new_polygon)

            massing = row[self.massing_col]
            new_massing = []
            for m in massing:
                new_m = {"id":m["id"], "massing":[]}
                for e in m["massing"]:
                    new_e = {"polygons":[[(round(p[0], n_digits), round(p[1], n_digits)) for p in polygon] for polygon in e["polygons"]], "bottom_elevation":e["bottom_elevation"], "top_elevation":e["top_elevation"]}
                    new_m["massing"].append(new_e)
                new_massing.append(new_m)
            return new_region, new_massing
        regions[[self.site_contour_col, self.massing_col]] = regions.progress_apply(lambda row: round_points(row), axis=1, result_type='expand')
        regions = regions[[self.regions_id_col, self.base_point_col, self.site_contour_col, self.building_properties_col, self.massing_col]]

        def is_valid(row):
            region = row[self.site_contour_col]
            region = [[(p[0], p[1]) for p in polygon] for polygon in region]
            try:
                region = self.polygons_to_shapely_converter(polygons=region)
                return region.is_valid
            except:
                return False
        regions["is_valid"] = regions.progress_apply(lambda row: is_valid(row), axis=1)
        regions = regions[regions["is_valid"]]
        
        return regions.to_dict("list")
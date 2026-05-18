from typing import List, Any, Dict
from collections import deque
import math
from copy import deepcopy
import json 

from tqdm import tqdm
import trimesh
import pandas as pd
from shapely.geometry import shape, GeometryCollection
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely import unary_union
import shapely
import osmium

from .dataset_creator import DatasetCreator
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter
from ..dataset_processor import GeoProjectProcessor
from ..dataset_merger import GeoPandasIdDatasetJoiner
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..mesh_compiler import MassingMeshCompiler

class LocationFilter:
    def node(self, n):
        return not n.location.valid()

    def way(self, w):
        return not all([n.location.valid() for n in w.nodes] )

class OSMBuildingsCreator(DatasetCreator):
    def __init__(self, *, osm_file_path: str,
                        tags_to_features_mapping_path: str,
                        features: List[str],
                        ops_eps: float,
                        user_eps: float,
                        floor_height: float,
                        int_features: List[str],
                        float_features: List[str],
                        height_feature: str,
                        min_height_feature: str,
                        max_height_feature: str,
                        n_floors_feature: str,
                        min_floor_feature: str,
                        max_floor_feature: str,
                        functions_feature: str,
                        functions_priority: List[str],
                        features_renaming: Dict[str, str],
                        filter_empty_features: List[str],
                        max_extrusions: int) -> None:
        self.osm_file_path = osm_file_path
        self.tags_to_features_mapping_path = tags_to_features_mapping_path
        self.features = features
        self.ops_eps = ops_eps
        self.user_eps = user_eps
        self.floor_height = floor_height
        self.int_features = int_features
        self.float_features = float_features
        self.height_feature = height_feature
        self.min_height_feature = min_height_feature
        self.max_height_feature = max_height_feature
        self.n_floors_feature = n_floors_feature
        self.min_floor_feature = min_floor_feature
        self.max_floor_feature = max_floor_feature
        self.functions_feature = functions_feature
        self.functions_priority = functions_priority
        self.features_renaming = features_renaming
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()
        self.mesh_creator = MassingMeshCompiler()
        self.filter_empty_features = filter_empty_features
        self.max_extrusions = max_extrusions

    def __call__(self) -> Dict[str, List[Any]]:
        tqdm.pandas()
        FEATURES = self.features

        #----------OPEN OSM FILE----------
        print("----------OPEN OSM FILE----------")

        path = self.osm_file_path
        objects = {"id":[], "type":[], "geometry":[], "tags":[], "nodes":[], "relations":[]}
        area_geometries = {}
        for i, o in enumerate(tqdm(osmium.FileProcessor(path).with_areas().with_filter(LocationFilter()).with_filter(osmium.filter.GeoInterfaceFilter(drop_invalid_geometries=False)))):
            if o.is_area():
                try:
                    area_geometries[str(o.orig_id())] = o.__geo_interface__['geometry']
                except:
                    continue
            else:
                obj = {
                    "id":str(o.id),
                    "type":o.type_str(),
                    "tags":dict(o.tags),
                    "nodes":[n.ref for n in o.nodes] if o.is_way() else [],
                    "relations":[{"role":m.role, "ref":m.ref, "type":m.type} for m in o.members] if o.is_relation() else []
                }
                try:
                    obj["geometry"] = o.__geo_interface__['geometry']
                except:
                    obj["geometry"] = None
                for k, v in obj.items():
                    objects[k].append(v)
        
        pd_objects = pd.DataFrame(objects)
        pd_objects["is_area"] = pd_objects.progress_apply(lambda row: True if row["id"] in area_geometries else False, axis=1)
        pd_objects["geometry"] = pd_objects.progress_apply(lambda row: area_geometries.get(row["id"], row["geometry"]), axis=1)

        pd_objects = pd_objects[["id", "type", "geometry", "tags", "nodes", "relations", "is_area"]]
        #have - geometry, relations, tags

        #----------GET BUILDING OBJECTS----------
        print("----------GET BUILDING OBJECTS----------")
        """
        Хотим отфильтровать конструктивные блоки зданий. 
        Конструктивным блоком считаем только замкнутые контуры.
        Утверждаем, что хорошими фичами являются только те, что принадлежат непосредственно конструктивным блокам.
        Следовательно, можем отфильтровать весь остальной OSM граф без потери информации о здании.
        """
        def is_building(row):
            if any([t.startswith("building") for t in row["tags"]]):
                return True
            if "type" in row["tags"]:
                if row["tags"]["type"].startswith("building"):
                    return True
            return False
        pd_building_objects = pd_objects[pd_objects.progress_apply(lambda row: is_building(row), axis=1)]

        pd_building_objects = pd_building_objects[pd_building_objects.progress_apply(lambda row: row["geometry"] != None, axis=1)]
        pd_building_objects = pd_building_objects[pd_building_objects["is_area"]]

        def collections_to_polygons(geometry):
            polygons = []
            for geom in geometry.geoms:
                if type(geom) == MultiPolygon:
                    polygons.extend(geom.geoms)
                elif type(geom) == GeometryCollection:
                    polygons.extend(collections_to_polygons(geom))
                elif type(geom) == Polygon:
                    polygons.append(geom)
            return polygons

        def get_polygons(row):
            geometry = shapely.make_valid(shape(row["geometry"]))
            if type(geometry) in [MultiPolygon, GeometryCollection]:
                polygons = collections_to_polygons(geometry)
                if len(polygons) == 0:
                    return None
                return polygons
            elif type(geometry) == Polygon:
                return [geometry]
            else:
                return None
        pd_building_objects["geometry"] = pd_building_objects.progress_apply(lambda row: get_polygons(row), axis=1)
        pd_building_objects = pd_building_objects[pd_building_objects.progress_apply(lambda row: row["geometry"] != None, axis=1)]

        blocks_dict = {"id":[], "type":[], "geometry":[], "tags":[], "nodes":[], "relations":[]}
        def get_blocks(row):
            for block in row["geometry"]:
                blocks_dict["geometry"].append(block)
                for key in blocks_dict.keys():
                    if key != "geometry":
                        blocks_dict[key].append(row[key])
        pd_building_objects.progress_apply(lambda row: get_blocks(row), axis=1)

        pd_building_blocks = pd.DataFrame(blocks_dict)
        pd_building_blocks["id"] = pd_building_blocks.apply(lambda row: str(row["id"]), axis=1)
        pd_building_blocks["id_buffer"] = pd_building_blocks["id"]
        pd_building_blocks = pd_building_blocks.set_index("id_buffer")

        geo_project_processor = GeoProjectProcessor(
            geo_col="geometry",
            from_format="EPSG:4326",
            to_format="EPSG:3857"
        )
        projected = geo_project_processor(dataset=pd_building_blocks.to_dict("list"))
        pd_building_blocks = pd.DataFrame(projected)

        pd_building_blocks["old_id"] = pd_building_blocks["id"]
        pd_building_blocks["id"] = [str(i) for i in range(len(pd_building_blocks))]
        old_to_new_ids_match = {}
        def get_ids_match(row):
            old_id = row["old_id"]
            if old_id not in old_to_new_ids_match:
                old_to_new_ids_match[old_id] = []
            old_to_new_ids_match[old_id].append(row["id"])
        pd_building_blocks.progress_apply(lambda row: get_ids_match(row), axis=1)

        pd_building_blocks = pd_building_blocks[["id", "geometry", "tags"]]

        pd_building_blocks["id_buffer"] = pd_building_blocks["id"]
        pd_building_blocks = pd_building_blocks.set_index("id_buffer")
        test = pd_building_blocks.loc[old_to_new_ids_match["13759061"]]
        print(len(test))
        #have - detached geometry objects, related to buildings

        #----------GET FEATURES----------
        print("----------GET FEATURES----------")

        path = self.tags_to_features_mapping_path
        with open(path, "r") as f:
            values_mapping = json.load(f)

        reverse_values_mapping = {}
        for group, features in values_mapping.items():
            for feature, tags in features.items():
                reverse_values_mapping[feature] = {}
                for tag, mapping in tags.items():
                    if tag == "DESCRIPTION":
                        continue
                    reverse_values_mapping[feature][tag] = {}

                    value_mapping = {}
                    dicts_to_parse = [(mapping, [])]
                    while len(dicts_to_parse) > 0:
                        cur_dict, cur_path = dicts_to_parse.pop(0)
                        for key, value in cur_dict.items():
                            if key == "BASE":
                                feature_value = "|".join(cur_path)
                                value_mapping[feature_value] = value
                            else:
                                if type(value) == list:
                                    feature_value = "|".join(cur_path + [key])
                                    value_mapping[feature_value] = value
                                elif type(value) == dict:
                                    dicts_to_parse.append((value, cur_path + [key]))

                    for feature_value, tag_values in value_mapping.items():
                        for value in tag_values:
                            reverse_values_mapping[feature][tag][value] = feature_value

        def get_features(row):
            row_features = {}
            for feature, tags in reverse_values_mapping.items():
                row_features[feature] = []
                for tag, mapping in tags.items():
                    if tag in row["tags"]:
                        if row["tags"][tag] in mapping:
                            row_features[feature].append(mapping[row["tags"][tag]])
            return row_features
        pd_building_blocks["features"] = pd_building_blocks.progress_apply(lambda row: get_features(row), axis=1)

        int_features = self.int_features
        float_features = self.float_features
        def process_numerical_features(row):
            for feature in row["features"]:
                if feature in int_features:
                    new_value = []
                    for v in row["features"][feature]:
                        try:
                            new_value.append(int(v))
                        except:
                            continue
                    row["features"][feature] = new_value
                elif feature in float_features:
                    new_value = []
                    for v in row["features"][feature]:
                        try:
                            new_value.append(float(v))
                        except:
                            continue
                    row["features"][feature] = new_value
        pd_building_blocks.progress_apply(lambda row: process_numerical_features(row), axis=1)

        pd_building_blocks = pd_building_blocks[["id", "geometry", "features"]]

        #----------GET SPATIAL GRAPH----------
        print("----------GET SPATIAL GRAPH----------")

        id_joiner = GeoPandasIdDatasetJoiner(
            join_dataset_key="join_dataset",
            join_dataset_geo_col="geometry",
            join_dataset_id_col="id",
            main_dataset_key="main_dataset",
            main_dataset_geo_col="geometry",
            predicate="intersects",
            output_col="spatial_neighbors"
        )
        pd_building_blocks = id_joiner(datasets={"main_dataset":pd_building_blocks.to_dict("list"), "join_dataset":pd_building_blocks.to_dict("list")})
        pd_building_blocks = pd.DataFrame(pd_building_blocks)
        pd_building_blocks["spatial_neighbors"] = pd_building_blocks.progress_apply(lambda row: [i for i in row["spatial_neighbors"] if i != row["id"]], axis=1)
        
        graph = {}
        def get_graph(row):
            graph[str(row["id"])] = [str(i) for i in row["spatial_neighbors"]]
        pd_building_blocks.progress_apply(lambda row: get_graph(row), axis=1)

        def find_components_bfs(n, adj_list):
            visited = {k: False for k in adj_list.keys()}
            components = []
            for i in tqdm(list(visited.keys())):
                if not visited[i]:
                    component = []
                    queue = deque([i])
                    visited[i] = True
                    while queue:
                        node = queue.popleft()
                        component.append(node)
                        for neighbor in adj_list[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                queue.append(neighbor)
                    components.append(component)
            return components
        components = find_components_bfs(len(graph), graph)
        node_to_component = {}
        for i, component in enumerate(components):
            for n in component:
                node_to_component[n] = str(i)

        pd_building_blocks["spatial_component"] = pd_building_blocks.progress_apply(lambda row: node_to_component[row["id"]], axis=1)

        pd_building_blocks = pd_building_blocks[["id", "geometry", "features", "spatial_neighbors", "spatial_component"]]

        #----------GET HEIGHTS----------
        print("----------GET HEIGHTS----------")

        floor_height = self.floor_height
        def get_feature_heights(row):
            heights = {}
            params_to_features = {
                "height": self.height_feature,
                "min_height": self.min_height_feature,
                "max_height": self.max_height_feature,
                "n_floors": self.n_floors_feature,
                "min_floor": self.min_floor_feature,
                "max_floor": self.max_floor_feature
            }
            for p in params_to_features:
                if len(row["features"][params_to_features[p]]) > 0:
                    heights[p] = row["features"][params_to_features[p]][0]

            if "height" in heights and "min_height" in heights and "max_height" not in heights:
                extrusion = {"bottom_elevation":heights["min_height"], "top_elevation":heights["min_height"] + heights["height"]}
            elif "height" not in heights and "min_height" in heights and "max_height" in heights:
                extrusion = {"bottom_elevation":heights["min_height"], "top_elevation":heights["max_height"]}
            elif "height" in heights and "min_height" not in heights and "max_height" in heights:
                extrusion = {"bottom_elevation":heights["max_height"] - heights["height"], "top_elevation":heights["max_height"]}
            elif "height" in heights and "min_height" not in heights and "max_height" not in heights:
                extrusion = {"bottom_elevation":0, "top_elevation":heights["height"]}
            elif "height" not in heights and "min_height" not in heights and "max_height" in heights:
                extrusion = {"bottom_elevation":0, "top_elevation":heights["max_height"]}

            elif "n_floors" in heights and "min_floor" in heights and "max_floor" not in heights:
                extrusion = {"bottom_elevation":heights["min_floor"]*floor_height, "top_elevation":(heights["min_floor"] + heights["n_floors"])*floor_height}
            elif "n_floors" not in heights and "min_floor" in heights and "max_floor" in heights:
                extrusion = {"bottom_elevation":heights["min_floor"]*floor_height, "top_elevation":heights["max_floor"]*floor_height}
            elif "n_floors" in heights and "min_floor" not in heights and "max_floor" in heights:
                extrusion = {"bottom_elevation":(heights["max_floor"] - heights["n_floors"])*floor_height, "top_elevation":heights["max_floor"]*floor_height}
            elif "n_floors" in heights and "min_floor" not in heights and "max_floor" not in heights:
                extrusion = {"bottom_elevation":0, "top_elevation":heights["n_floors"]*floor_height}
            elif "n_floors" not in heights and "min_floor" not in heights and "max_floor" in heights:
                extrusion = {"bottom_elevation":0, "top_elevation":heights["max_floor"]*floor_height}

            else:
                extrusion = None
            return extrusion
        pd_building_blocks["feature_heights"] = pd_building_blocks.progress_apply(lambda row: get_feature_heights(row), axis=1)

        feature_heights = {}
        def extract_feature_heights(row):
            feature_heights[row["id"]] = row["feature_heights"]
        pd_building_blocks.progress_apply(lambda row: extract_feature_heights(row), axis=1)

        areas = {}
        def extract_areas(row):
            areas[row["id"]] = row["geometry"].area
        pd_building_blocks.progress_apply(lambda row: extract_areas(row), axis=1)

        components = {}
        def extract_components(row):
            if row["spatial_component"] not in components:
                components[row["spatial_component"]] = []
            components[row["spatial_component"]].append(row["id"])
        pd_building_blocks.progress_apply(lambda row: extract_components(row), axis=1)

        def get_area_heights(members):
            max_area = 0
            max_heights = None
            for ind in members:
                heights = feature_heights[ind]
                area = areas[ind]
                if heights != None and area > max_area:
                    max_area = area
                    max_heights = heights
            return max_heights

        function_height = {}
        def get_function_height(row):
            base_functions = list(set([f.split("|")[0] for f in row["features"][self.functions_feature]]))
            heights = row["feature_heights"]
            if heights == None:
                return
            for f in base_functions:
                if f not in function_height:
                    function_height[f] = []
                function_height[f].append(heights)
        pd_building_blocks.progress_apply(lambda row: get_function_height(row), axis=1)
        for f in function_height:
            function_height[f] = np.median([e["top_elevation"] for e in function_height[f]])

        functions_priority = self.functions_priority
        def get_heights(row):
            heights = row["feature_heights"]
            if heights == None:
                members = row["spatial_neighbors"]
                heights = get_area_heights(members)
            if heights == None:
                members = components[row["spatial_component"]]
                heights = get_area_heights(members)
            if heights == None:
                base_functions = list(set([f.split("|")[0] for f in row["features"]["building_function"]]))
                for f in functions_priority:
                    if f in base_functions:
                        heights = {"bottom_elevation":0, "top_elevation":function_height[f]}
            if heights == None:
                heights = {"bottom_elevation":0, "top_elevation":function_height["residential"]}
            if heights["bottom_elevation"] > heights["top_elevation"]:
                bottom_elevation = heights["bottom_elevation"]
                top_elevation = heights["top_elevation"]
                heights["bottom_elevation"] = top_elevation
                heights["top_elevation"] = bottom_elevation
            if heights["bottom_elevation"] == heights["top_elevation"]:
                return None
            return heights
        pd_building_blocks["heights"] = pd_building_blocks.progress_apply(lambda row: get_heights(row), axis=1)
        pd_building_blocks = pd_building_blocks[pd_building_blocks.progress_apply(lambda row: row["heights"] != None, axis=1)]

        def simplify_elevations(row):
            heights = row["heights"]
            n_digits = int(abs(math.log10(self.user_eps)))
            new_e = deepcopy(heights)
            new_e["bottom_elevation"] = round(heights["bottom_elevation"], n_digits)
            if new_e["bottom_elevation"] == round(new_e["bottom_elevation"]):
                new_e["bottom_elevation"] = int(new_e["bottom_elevation"])
            new_e["top_elevation"] = round(heights["top_elevation"], n_digits)
            if new_e["top_elevation"] == round(new_e["top_elevation"]):
                new_e["top_elevation"] = int(new_e["top_elevation"])
            return new_e
        pd_building_blocks["heights"] = pd_building_blocks.progress_apply(lambda row: simplify_elevations(row), axis=1)

        pd_building_blocks = pd_building_blocks[["id", "geometry", "features", "spatial_neighbors", "spatial_component", "heights"]]

        pd_building_blocks["id_buffer"] = pd_building_blocks["id"]
        pd_building_blocks = pd_building_blocks.set_index("id_buffer")
        test = pd_building_blocks.loc[old_to_new_ids_match["13759061"]]
        print(len(test))

        #----------GET EXTRUSIONS----------
        print("----------GET EXTRUSIONS----------")

        def get_extrusion(row):
            extrusion = {
                "polygons":row["geometry"],
                "bottom_elevation":row["heights"]["bottom_elevation"],
                "top_elevation":row["heights"]["top_elevation"],
            }
            return extrusion
        pd_building_blocks["extrusion"] = pd_building_blocks.progress_apply(lambda row: get_extrusion(row), axis=1)

        pd_building_blocks = pd_building_blocks[["id", "extrusion", "features", "spatial_neighbors", "spatial_component"]]

        #----------GET BUILDINGS----------
        print("----------GET BUILDINGS----------")

        buildings_dict = {}
        def get_massings(row):
            if row["spatial_component"] not in buildings_dict:
                buildings_dict[row["spatial_component"]] = {"building":{"id": "0", "massing":[]}, "features":{f:[] for f in FEATURES}}
            buildings_dict[row["spatial_component"]]["building"]["massing"].append(row["extrusion"])
            for f in FEATURES:
                if f in row["features"]:
                    buildings_dict[row["spatial_component"]]["features"][f].extend(row["features"][f])
        pd_building_blocks.progress_apply(lambda row: get_massings(row), axis=1)

        for key, value in buildings_dict.items():
            for f in value["features"]:
                value["features"][f] = list(set(value["features"][f]))

        buildings_dataset = []
        for key in tqdm(list(buildings_dict.keys())):
            sample = {"id":str(key)}
            sample.update(buildings_dict[key])
            buildings_dataset.append(sample)
        buildings_dataset = pd.DataFrame(buildings_dataset)
        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")

        def filter_empty_features(row):
            for f in self.filter_empty_features:
                if len(row["features"][f]) == 0:
                    return False
            return True
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: filter_empty_features(row), axis=1)]

        def rename_features(row):
            new_features = {}
            for f in row["features"]:
                if f in self.features_renaming:
                    new_features[self.features_renaming[f]] = row["features"][f]
                else:
                    new_features[f] = row["features"][f]
            return new_features
        buildings_dataset["features"] = buildings_dataset.progress_apply(lambda row: rename_features(row), axis=1)

        buildings_dataset = buildings_dataset[["id", "building", "features"]]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        #----------FIX BUIDINGS----------
        print("----------FIX BUIDINGS----------")

        def get_mesh(row):
            try:
                building = {"id":row["building"]["id"], "massing":[]}
                for e in row["building"]["massing"]:
                    new_e = {
                        "polygons":self.shapely_to_polygons_converter(polygons=e["polygons"]),
                        "bottom_elevation":e["bottom_elevation"],
                        "top_elevation":e["top_elevation"],
                    }
                    building["massing"].append(new_e)
                return self.mesh_creator(obj=[building])
            except:
                return None
        
        def get_intersection_volume(e1, e2):
            # Получаем shapely-полигоны
            p1 = self.polygons_to_shapely_converter(polygons=[[(point[0], point[1]) for point in polygon] for polygon in e1["polygons"]])
            p2 = self.polygons_to_shapely_converter(polygons=[[(point[0], point[1]) for point in polygon] for polygon in e2["polygons"]])

            try:
                intersection = p1.intersection(p2)
            except Exception:
                return 0.0

            # Определение пересечения по высоте
            z1_bottom = e1["bottom_elevation"]
            z1_top = e1["top_elevation"]
            z2_bottom = e2["bottom_elevation"]
            z2_top = e2["top_elevation"]

            # Находим пересечение по высоте
            z_overlap_bottom = max(z1_bottom, z2_bottom)
            z_overlap_top = min(z1_top, z2_top)
            dz = z_overlap_top - z_overlap_bottom

            # Если пересечений по высоте нет или по площади нет - объема нет
            if dz <= 0 or intersection.is_empty:
                return 0.0

            intersection_area = intersection.area

            # ---- Разбор по вариантам, если требуется подробный вывод (оставим как комментарии)
            # elevation_case = 
            #   0  - e1 снизу заходит внутрь e2 и выходит сверху e2
            #   1  - e1 снизу под e2, но верх заходит в e2
            #   2  - e2 полностью внутри e1
            #   3  - e1 полностью внутри e2
            #   4  - e1 и e2 не пересекаются по высоте

            # polygon_case =
            #   0 - совпадает с p1 (площадь пересечения примерно равна p1)
            #   1 - совпадает с p2 (площадь пересечения примерно равна p2)
            #   2 - частичное пересечение (площадь есть, но не совпадает ни с p1 ни с p2)
            #   3 - нет пересечения

            # Если площадь пересечения очень маленькая - возвращаем 0
            if intersection_area < self.ops_eps:
                return 0.0

            # Возвращаем объем = площадь пересечения * высота пересечения
            return intersection_area * dz

        def get_volume(e):
            return e["polygons"].area * (e["top_elevation"] - e["bottom_elevation"])

        def get_volume_components_count(building):
            massing = building["massing"]
            eps = self.ops_eps
            if len(massing) == 0:
                return 0

            graph = {i: [] for i in range(len(massing))}
            for i in range(len(massing)):
                e1 = massing[i]
                for j in range(i + 1, len(massing)):
                    e2 = massing[j]
                    bottom = max(e1["bottom_elevation"], e2["bottom_elevation"])
                    top = min(e1["top_elevation"], e2["top_elevation"])
                    if top - bottom < -eps:
                        continue

                    if not e1["polygons"].intersects(e2["polygons"]):
                        continue

                    graph[i].append(j)
                    graph[j].append(i)

            visited = set()
            n_components = 0
            for i in range(len(massing)):
                if i in visited:
                    continue
                n_components += 1
                queue = deque([i])
                visited.add(i)
                while len(queue) > 0:
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return n_components

        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: len(row["building"]["massing"]) <= self.max_extrusions, axis=1)]

        def select_multiple_volumes(row):
            return get_volume_components_count(row["building"]) > 1
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not select_multiple_volumes(row), axis=1)]
        
        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))
        
        def remove_outline_extrusions(row):
            """
            Функция берет набор экструзий, определенных полигоном и отметкой верха и низа.
            Экструзии - OSM объекты, которые могут определять или не определять реальную геометрию здания.
            Объединение полигонов экструзий гарантированно дает неделимую площадь.
            Функция удаляет экструзии таким образом, чтобы сохранилось как можно больше геометрии здания.
            Алгоритм:
            1. Определяем экструзии, защищенные от удаления:
                1.1 Делаем объединение всех полигонов
                1.2 Для каждой экструзии делаем объединение всех полигонов без этой экструзии
                1.3 Если объединение без экструзии меньше по площади, то помечаем экструзию как защищенную от удаления
                1.4 Если объединение без экструзии распадается на объемные компоненты связности, то помечаем экструзию как защищенную от удаления
                1.5 Для каждой экструзии проверяем - если ее bottom_elevation > 0, то защищаем ее от удаления
            2. Определяем финальные экструзии для удаления:
                2.1 Берем все экструзии, которые не помечены как защищенные от удаления и не помечены как удаленные
                2.2 Для каждой такой экструзии берем все экструзии, с которыми она пересекается по полигону
                При этом пересечением считается только пересечение по площади, а не касание
                При этом не берем экструзии, которые помечены как удаленные
                При этом контур экструзии либо полностью является внешним для объединения, либо частично, 
                либо вообще не является
                2.3 Проверяем, что при удалении экструзии не изменится контур объединения (группы!). Если изменится, то пропускаем
                2.4 Если контур полностью является внешним:
                    2.4.1 Если у всех экструзий объединения одна высота, то помечаем экструзию как удаленную
                    2.4.2 Если две и более уникальных высот:
                        2.4.3.1 Если top_elevation экструзии равна максимальной top_elevation 
                        всех остальных экструзий из объединения, то удаляем экструзию
                2.5 Если контур частично является внешним или не является им вовсе:
                    2.5.1 Делаем объединение пересечений всех экструзий объединения с этой, получаем объединение, 
                    где эта экструзия является внешней
                    2.5.2 Если у всех экструзий объединения одна высота, то помечаем экструзию как удаленную
                    2.5.3 Если две и более уникальных высот:
                        2.5.3.1 Если top_elevation экструзии равна максимальной top_elevation 
                        всех остальных экструзий из объединения, то удаляем экструзию
            3. Удаляем экструзии, помеченные как удаленные

            """
            AREA_EPS = self.ops_eps
            bad_result = None

            building = deepcopy(row["building"])
            extrusions = building["massing"]
            n = len(extrusions)
            # Служебные словари
            protected = set()   # индексы экструзий, которые нельзя удалить
            to_remove = set()   # индексы экструзий, которые нужно удалить

            # 1. Поиск защищённых экструзий
            # 1.1 Объединяем все полигоны
            all_polys = [e["polygons"] for e in extrusions]
            try:
                full_union = unary_union(all_polys)
            except:
                return bad_result
            full_area = full_union.area

            for i, ext in enumerate(extrusions):
                # 1.2 / 1.3 Поочередно исключаем i-й и смотрим, насколько уменьшилась площадь объединения
                rest_polys = [extrusions[j]["polygons"] for j in range(n) if j != i]
                try:
                    rest_union = unary_union(rest_polys)
                except:
                    return bad_result
                rest_area = rest_union.area
                if rest_area < full_area - AREA_EPS:
                    protected.add(i)
                    continue

                # 1.4 Проверяем, распадается ли объединение без экструзии на объемные компоненты связности
                rest_building = {"id":building["id"], "massing":[extrusions[j] for j in range(n) if j != i]}
                rest_volume = get_volume_components_count(rest_building)
                if rest_volume > 1:
                    protected.add(i)
                    continue

                # 1.5 Проверяем отметку низа
                if ext["bottom_elevation"] > 0:
                    protected.add(i)

            # 2. Определяем экструзии-кандидаты на удаление
            candidates = set(range(n)) - protected

            # Дополнительные данные для проверки пересечений
            poly_objs = [e["polygons"] for e in extrusions]

            for i in candidates:
                # Пересекающиеся экструзии (только по площади, и которые ещё не удалены)
                overlaps = []
                for j in range(n):
                    if j == i or j in to_remove:
                        continue
                    inter = poly_objs[i].intersection(poly_objs[j])
                    if not inter.is_empty and inter.area > AREA_EPS:
                        overlaps.append(j)

                # Группа — текущая экструзия и пересекающиеся с ней (по площади)
                group = [i] + overlaps

                # 2.3 Проверяем — изменится ли контур объединения только внутри группы экструзий при удалении i
                # То есть, работаем только с group, а не со всеми экструдерами!
                polys_in_group = [poly_objs[j] for j in group]
                try:
                    union_group = unary_union(polys_in_group)
                except:
                    return bad_result

                group_without_i = [poly_objs[j] for j in group if j != i]
                if not group_without_i:
                    continue  # Нечего анализировать (группа из одной экструзии)
                try:
                    union_without_i = unary_union(group_without_i)
                except:
                    return bad_result

                # Если многоугольник группы без i не совпадает с изначальным (для группы) — нельзя удалять
                try:
                    if not union_without_i.equals_exact(union_group, tolerance=AREA_EPS):
                        continue
                except:
                    return bad_result

                # 2.4.1 и 2.5.2: группа по высотам
                heights = set([extrusions[j]["top_elevation"] for j in group])
                if len(heights) == 1:
                    to_remove.add(i)
                    continue

                # 2.4.3.1 и 2.5.3.1: top_elevation равен max остальных top_elevation (кроме i)
                others = [extrusions[j]["top_elevation"] for j in group if j != i]
                if len(others) > 0 and extrusions[i]["top_elevation"] >= max(others):
                    to_remove.add(i)
                    continue

            # 3. Возвращаем здание без удалённых экструзий
            new_massings = [mass for idx, mass in enumerate(building["massing"]) if idx not in to_remove]
            building["massing"] = new_massings
            return building
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: remove_outline_extrusions(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: row["building"] != None, axis=1)]

        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not select_multiple_volumes(row), axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def remove_intersection_volumes(row):
            bad_result = None
            building = row["building"]
            massing = building["massing"]
            eps = self.ops_eps

            buffer = {i:[massing[i]] for i in range(len(massing))}
            for i in range(len(massing)):
                for j in range(i + 1, len(massing)):
                    e2 = massing[j]
                    new_es = []
                    for e1 in buffer[i]:
                        p1 = e1["polygons"]
                        p2 = e2["polygons"]
                        bottom = max(e1["bottom_elevation"], e2["bottom_elevation"])
                        top = min(e1["top_elevation"], e2["top_elevation"])
                        try:
                            intersection = p1.intersection(p2)
                            difference = p1.difference(p2)
                        except:
                            return bad_result

                        if top - bottom <= eps or shapely.area(intersection) <= eps:
                            new_es.append(e1)
                            continue

                        if e1["bottom_elevation"] < bottom:
                            new_es.append({
                                "polygons":p1,
                                "bottom_elevation":e1["bottom_elevation"],
                                "top_elevation":bottom
                            })
                        if top < e1["top_elevation"]:
                            new_es.append({
                                "polygons":p1,
                                "bottom_elevation":top,
                                "top_elevation":e1["top_elevation"]
                            })
                        if not difference.is_empty and shapely.area(difference) > eps:
                            if type(difference) == Polygon:
                                polygons = [difference]
                            elif type(difference) == MultiPolygon:
                                polygons = list(difference.geoms)
                            else:
                                polygons = []
                            for p in polygons:
                                new_es.append({
                                    "polygons":p,
                                    "bottom_elevation":bottom,
                                    "top_elevation":top
                                })
                    buffer[i] = new_es
            massing = []
            for es in buffer.values():
                massing += es

            return {"id":building["id"], "massing":massing}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: remove_intersection_volumes(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: row["building"] != None, axis=1)]
        
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not select_multiple_volumes(row), axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def sync_intersection_faces(row):
            bad_result = None
            building = row["building"]
            massing = building["massing"]

            graph = {i:[] for i in range(len(massing))}
            for i in range(len(massing)):
                e1 = massing[i]
                for j in range(i + 1, len(massing)):
                    e2 = massing[j]
                    if (e1["top_elevation"] < e2["bottom_elevation"] or 
                        e1["bottom_elevation"] > e2["top_elevation"]):
                        continue
                    try:
                        if e1["polygons"].boundary.intersects(e2["polygons"].boundary):
                            graph[i].append(j)
                            graph[j].append(i)
                    except:
                        return bad_result

            groups = []
            used = []
            for i in range(len(massing)):
                if i in used:
                    continue
                group = []
                queue = deque([i])
                used.append(i)
                while len(queue) > 0:
                    idx = queue.popleft()
                    group.append(idx)
                    for j in graph[idx]:
                        if j not in used:
                            used.append(j)
                            queue.append(j)
                groups.append(group)

            synced = [None for _ in range(len(massing))]
            for group in groups:
                linework = unary_union([massing[i]["polygons"].boundary for i in group])
                for i in group:
                    e1 = massing[i]
                    p = e1["polygons"]
                    try:
                        polygons = [g for g in shapely.get_parts(shapely.polygonize([linework])) if p.covers(g)]
                        if len(polygons) > 0:
                            p = unary_union(polygons)
                    except:
                        return bad_result
                    synced[i] = {
                        "polygons":p,
                        "bottom_elevation":e1["bottom_elevation"],
                        "top_elevation":e1["top_elevation"]
                    }

            return {"id":building["id"], "massing":synced}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: sync_intersection_faces(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: row["building"] != None, axis=1)]
        
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not select_multiple_volumes(row), axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def merge_intersection_extrusions(row):
            bad_result = None
            building = row["building"]
            massing = building["massing"]
            eps = self.ops_eps

            merged = []
            buffer = {i:massing[i] for i in range(len(massing))}
            for i in range(len(massing)):
                for j in range(len(massing)):
                    if i == j or j in merged:
                        continue
                    e1 = buffer[i]
                    e2 = buffer[j]
                    if (e1["bottom_elevation"] != e2["bottom_elevation"] or 
                        e1["top_elevation"] != e2["top_elevation"]):
                        continue
                    try:
                        if shapely.area(e1["polygons"].intersection(e2["polygons"])) > eps:
                            buffer[i] = {
                                "polygons":e1["polygons"].union(e2["polygons"]),
                                "bottom_elevation":e1["bottom_elevation"],
                                "top_elevation":e1["top_elevation"]
                            }
                            merged.append(j)
                    except:
                        return bad_result

            return {"id":building["id"], "massing":[v for i, v in buffer.items() if i not in merged]}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: merge_intersection_extrusions(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: row["building"] != None, axis=1)]
        
        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not select_multiple_volumes(row), axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def simplify_points(row):
            bad_result = None
            building = row["building"]
            massing = building["massing"]
            eps = self.user_eps
            n_digits = int(abs(math.log10(eps)))
       
            new_massing = []

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

            for e in massing:
                polygons = []
                if type(e["polygons"]) == Polygon:
                    polygons = [e["polygons"]]
                elif type(e["polygons"]) == MultiPolygon:
                    polygons = list(e["polygons"].geoms)

                new_polygons = []
                for p in polygons:
                    try:
                        exterior = simplify_ring(list(p.exterior.coords))
                        interiors = [simplify_ring(list(r.coords)) for r in p.interiors]
                        interiors = [r for r in interiors if len(r) >= 4]
                        new_p = Polygon(exterior, interiors) if len(interiors) > 0 else Polygon(exterior)
                        if not new_p.is_empty and shapely.area(new_p) > eps**2 and shapely.is_valid(new_p):
                            new_polygons.append(new_p)
                    except:
                        return bad_result

                if len(new_polygons) == 0:
                    continue
                new_e = deepcopy(e)
                new_e["polygons"] = unary_union(new_polygons)
                new_massing.append(new_e)
            if len(new_massing) == 0:
                return bad_result
            return {"id":building["id"], "massing":new_massing}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: simplify_points(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: row["building"] != None, axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def select_empty_buildings(row):
            building = row["building"]
            massing = building["massing"]
            if len(massing) == 0:
                return True
            for e in massing:
                if shapely.area(e["polygons"]) < self.ops_eps:
                    return True
            return False
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not select_empty_buildings(row), axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def shift_bottom_to_zero(row):
            building = row["building"]
            massing = building["massing"]
            min_bottom = min([e["bottom_elevation"] for e in massing])
            new_massing = []
            for e in massing:
                new_e = deepcopy(e)
                new_e["bottom_elevation"] = e["bottom_elevation"] - min_bottom
                new_e["top_elevation"] = e["top_elevation"] - min_bottom
                new_massing.append(new_e)
            return {"id":building["id"], "massing":new_massing}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: shift_bottom_to_zero(row), axis=1)

        def sort_by_elevations(row):
            building = row["building"]
            massing = building["massing"]
            massing.sort(key=lambda x: x["bottom_elevation"])
            return {"id":building["id"], "massing":massing}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: sort_by_elevations(row), axis=1)
        
        #----------FINAL FILTERING----------
        print("----------FINAL FILTERING----------")
        
        def is_multiple_polygons(row):
            building = row["building"]
            massing = building["massing"]
            for e in massing:
                polygons = e["polygons"]
                if type(polygons) == MultiPolygon:
                    return True
            return False
        buildings_dataset["is_multiple_polygons"] = buildings_dataset.progress_apply(lambda row: is_multiple_polygons(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not is_multiple_polygons(row), axis=1)]

        def is_multiple_volumes(row):
            building = row["building"]
            if get_volume_components_count(building) > 1:
                return True
            return False
        buildings_dataset["is_multiple_volumes"] = buildings_dataset.progress_apply(lambda row: is_multiple_volumes(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not is_multiple_volumes(row), axis=1)]

        def is_not_same_height_intersect(row):
            m = row["building"]
            for i in range(len(m["massing"])):
                for j in range(i + 1, len(m["massing"])):
                    e1 = m["massing"][i]
                    e2 = m["massing"][j]
                    p1 = e1["polygons"]
                    p2 = e2["polygons"]

                    if e1["bottom_elevation"] != e2["bottom_elevation"] or e1["top_elevation"] != e2["top_elevation"]:
                        continue
                    else:
                        try:
                            intersection = p1.intersection(p2)
                        except:
                            return True
                        if shapely.area(intersection) > self.ops_eps:
                            return False
            return True
        buildings_dataset["is_not_same_height_intersect"] = buildings_dataset.progress_apply(lambda row: is_not_same_height_intersect(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: is_not_same_height_intersect(row), axis=1)]

        def is_extrusion_can_cut(row):
            m = row["building"]
            for i in range(len(m["massing"])):
                for j in range(i + 1, len(m["massing"])):
                    e1 = m["massing"][i]
                    e2 = m["massing"][j]
                    p1 = e1["polygons"]
                    p2 = e2["polygons"]

                    try:
                        intersection = p1.intersection(p2)
                        difference = p1.difference(p2)
                    except:
                        return False

                    elevation_case = None
                    if (e1["bottom_elevation"] >= e2["bottom_elevation"] and 
                        e1["bottom_elevation"] < e2["top_elevation"] and 
                        e1["top_elevation"] > e2["top_elevation"]):
                        elevation_case = 0
                    elif (e1["bottom_elevation"] < e2["bottom_elevation"] and 
                        e1["top_elevation"] > e2["bottom_elevation"] and 
                        e1["top_elevation"] <= e2["top_elevation"]):
                        elevation_case = 1
                    elif (e2["bottom_elevation"] > e1["bottom_elevation"] and 
                        e2["top_elevation"] < e1["top_elevation"]):
                        elevation_case = 2
                    elif (e1["bottom_elevation"] > e2["bottom_elevation"] and 
                        e1["top_elevation"] < e2["top_elevation"]):
                        elevation_case = 3
                    elif (e1["bottom_elevation"] > e2["top_elevation"] or 
                        e1["top_elevation"] < e2["bottom_elevation"]):
                        elevation_case = 4
                    
                    polygon_case = None
                    if abs(shapely.area(intersection) - shapely.area(p1)) < 1e-4:
                        polygon_case = 0
                    elif shapely.area(intersection) == 0:
                        polygon_case = 3
                    else:
                        if type(difference) == Polygon:
                            polygon_case = 1
                        elif type(difference) == MultiPolygon:
                            polygon_case = 2
                        else:
                            continue
                    
                    if not (polygon_case == 0 and
                        (elevation_case == 0 or elevation_case == 1)):
                        continue
                    return True
            return False
        buildings_dataset["is_extrusion_can_cut"] = buildings_dataset.progress_apply(lambda row: is_extrusion_can_cut(row), axis=1)
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: not is_extrusion_can_cut(row), axis=1)]

        buildings_dataset["trimesh_building"] = buildings_dataset.progress_apply(lambda row: get_mesh(row), axis=1)
        
        def is_mesh_valid(row):
            if row["trimesh_building"] != None:
                return True
            else:
                return False
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: is_mesh_valid(row), axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def is_watertight(row):
            building = row["trimesh_building"]
            if not building.is_watertight:
                return False
            return True
        buildings_dataset = buildings_dataset[buildings_dataset.progress_apply(lambda row: is_watertight(row), axis=1)]

        buildings_dataset["id_buffer"] = buildings_dataset["id"]
        buildings_dataset = buildings_dataset.set_index("id_buffer")
        components = [node_to_component[i] for i in old_to_new_ids_match["13759061"]]
        test = buildings_dataset.loc[components]
        print(len(test))

        def convert_polygons(row):
            building = row["building"]
            massing = building["massing"]
            new_massing = []
            for e in massing:
                new_e = deepcopy(e)
                new_e["polygons"] = self.shapely_to_polygons_converter(polygons=e["polygons"])
                new_massing.append(new_e)
            return {"id":building["id"], "massing":new_massing}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: convert_polygons(row), axis=1)

        def round_points(row):
            building = row["building"]
            massing = building["massing"]
            n_digits = int(abs(math.log10(self.user_eps)))
            new_massing = []
            for e in massing:
                new_e = deepcopy(e)
                new_e["polygons"] = [[(round(p[0], n_digits), round(p[1], n_digits)) for p in polygon] for polygon in e["polygons"]]
                new_massing.append(new_e)
            return {"id":building["id"], "massing":new_massing}
        buildings_dataset["building"] = buildings_dataset.progress_apply(lambda row: round_points(row), axis=1)
        
        buildings_dataset = buildings_dataset[["id", "building", "features"]]

        dataset = buildings_dataset.to_dict("list")
        return dataset
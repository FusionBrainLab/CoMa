from typing import List, Any, Dict
import math

import shapely
from tqdm import tqdm
from shapely.ops import unary_union, polygonize
import pandas as pd
import geopandas as gpd
import numpy as np
import pyrosm
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry import shape
import osmium
from shapely import set_precision

from .dataset_creator import DatasetCreator
from ..dataset_processor import GeoProjectProcessor
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

def tag_value_equals(value, expected_value):
    """OSM tags can be scalar values or lists; compare both forms uniformly."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return any(tag_value_equals(item, expected_value) for item in value)
    if pd.isna(value):
        return expected_value is None
    return str(value).strip().lower() == str(expected_value).strip().lower()

class LocationFilter:
    def node(self, n):
        return not n.location.valid()

    def way(self, w):
        return not all([n.location.valid() for n in w.nodes] )


class OSMRegionsCreator(DatasetCreator):
    def __init__(self, *, osm_file_path: str,
                        boundary_line_types: List[str],
                        disabled_tag_values: List[Any],
                        ground_layer_values: List[Any],
                        min_area: float,
                        user_eps: float,
                        min_circularity: float) -> None:
        self.osm_file_path = osm_file_path
        self.boundary_line_types = boundary_line_types
        self.disabled_tag_values = disabled_tag_values
        self.ground_layer_values = ground_layer_values
        self.min_area = min_area
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.user_eps = user_eps
        self.min_circularity = min_circularity

    def __call__(self) -> Dict[str, List[Any]]:
        tqdm.pandas()
        path = self.osm_file_path
        
        #----------LOAD NETWORK----------
        print("----------LOAD NETWORK----------")
        osm = pyrosm.OSM(path)
        nodes_gdf, edges_gdf = osm.get_network(nodes=True, network_type='all')
        print(f"Loaded {len(edges_gdf)} edges and {len(nodes_gdf)} nodes")

        #----------EXTRACT ROADS----------
        print("----------EXTRACT ROADS----------")
        boundary_line_types = self.boundary_line_types
        disabled_tag_values = self.disabled_tag_values
        ground_layer_values = self.ground_layer_values

        masks = []
        if 'highway' in edges_gdf.columns:
            # Roads are explicit region boundaries; keep only road classes that split urban blocks.
            masks.append(edges_gdf['highway'].apply(
                lambda value: any(tag_value_equals(value, line_type) for line_type in boundary_line_types)
            ))
        if 'waterway' in edges_gdf.columns:
            # Waterways have many possible values, so here we keep any non-empty water barrier.
            masks.append(edges_gdf['waterway'].apply(
                lambda value: not any(tag_value_equals(value, disabled_value) for disabled_value in disabled_tag_values)
            ))

        if masks:
            boundary_mask = masks[0]
            for mask in masks[1:]:
                boundary_mask = boundary_mask | mask
        else:
            raise ValueError("OSM network has no highway or waterway columns to build region boundaries")

        # Region polygons require a planar ground-level graph: bridges, tunnels and non-zero layers
        # geometrically cross other roads, but should not split land parcels at street level.
        ground_mask = pd.Series(True, index=edges_gdf.index)
        if 'bridge' in edges_gdf.columns:
            ground_mask = ground_mask & edges_gdf['bridge'].apply(
                lambda value: any(tag_value_equals(value, disabled_value) for disabled_value in disabled_tag_values)
            )
        if 'tunnel' in edges_gdf.columns:
            ground_mask = ground_mask & edges_gdf['tunnel'].apply(
                lambda value: any(tag_value_equals(value, disabled_value) for disabled_value in disabled_tag_values)
            )
        if 'layer' in edges_gdf.columns:
            ground_mask = ground_mask & edges_gdf['layer'].apply(
                lambda value: any(tag_value_equals(value, ground_layer) for ground_layer in ground_layer_values)
            )

        lines_gdf = edges_gdf[boundary_mask & ground_mask].copy()
        lines_gdf = lines_gdf[lines_gdf.geometry.notna() & ~lines_gdf.geometry.is_empty]
        print(f"Selected {len(lines_gdf)} ground boundary lines")
        if lines_gdf.empty:
            raise ValueError("No valid ground boundary lines found to build region polygons")

        #----------LOAD AREAS----------
        print("----------LOAD AREAS----------")

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
        areas = pd_objects[pd_objects.progress_apply(lambda row: row["is_area"], axis=1)]
        def is_building(row):
            if any([t.startswith("building") for t in row["tags"]]):
                return True
            if "type" in row["tags"]:
                if row["tags"]["type"].startswith("building"):
                    return True
            return False
        areas = areas[areas.progress_apply(lambda row: not is_building(row), axis=1)]

        def is_region(row):
            region_tags = ["amenity", "boundary", "landuse", "leisure", "place"]
            if any([t.startswith(tag) for tag in region_tags for t in row["tags"]]):
                return True
            else:
                return False
        areas = areas[areas.progress_apply(lambda row: is_region(row), axis=1)]

        areas = areas[["id", "type", "geometry", "tags", "nodes", "relations", "is_area"]]

        #----------PREPARE ROADS FOR POLYGONIZATION----------
        print("----------PREPARE ROADS FOR POLYGONIZATION----------")
        geo_project_processor = GeoProjectProcessor(
            geo_col="geometry",
            from_format="EPSG:4326",
            to_format="EPSG:3857"
        )
        projected_lines = geo_project_processor(dataset=lines_gdf.to_dict("list"))
        lines_gdf = gpd.GeoDataFrame(pd.DataFrame(projected_lines), geometry="geometry", crs="EPSG:3857")

        # Do not simplify here: short streets and small links can be real splitting edges.
        line_geometries = [
            make_valid(geometry)
            for geometry in tqdm(lines_gdf.geometry, total=len(lines_gdf), desc="Preparing linework")
            if geometry is not None and not geometry.is_empty
        ]
        if not line_geometries:
            raise ValueError("No valid line geometries found after projection")

        #----------PREPARE AREAS FOR POLYGONIZATION----------
        print("----------PREPARE AREAS FOR POLYGONIZATION----------")
        areas_for_linework = areas[areas["geometry"].notna()].copy()
        areas_for_linework["geometry"] = areas_for_linework.progress_apply(
            lambda row: shapely.make_valid(shape(row["geometry"]), method='structure', keep_collapsed=False),
            axis=1
        )
        projected_areas = geo_project_processor(dataset=areas_for_linework.to_dict("list"))
        areas_gdf = gpd.GeoDataFrame(pd.DataFrame(projected_areas), geometry="geometry", crs="EPSG:3857")

        def get_area_boundaries(geometry):
            boundaries = []
            if geometry is None or geometry.is_empty:
                return boundaries
            if type(geometry) == Polygon:
                boundaries.append(geometry.boundary)
            elif type(geometry) == MultiPolygon:
                boundaries.extend([g.boundary for g in geometry.geoms])
            elif type(geometry) == GeometryCollection:
                for g in geometry.geoms:
                    boundaries.extend(get_area_boundaries(g))
            return boundaries

        area_boundaries = []
        for geometry in tqdm(areas_gdf.geometry, total=len(areas_gdf), desc="Preparing area boundaries"):
            try:
                area_boundaries.extend(get_area_boundaries(geometry))
            except:
                continue
        line_geometries.extend(area_boundaries)
        print(f"Added {len(area_boundaries)} area boundary lines")

        #----------POLYGONIZE ROADS AND AREAS----------
        print("----------POLYGONIZE ROADS AND AREAS----------")

        # unary_union nodes intersections; polygonize then returns the minimal bounded faces.
        new_line_geometries = []
        for geometry in tqdm(line_geometries):
            new_geometry = set_precision(geometry, self.user_eps)
            new_line_geometries.append(new_geometry)
        line_geometries = new_line_geometries

        linework = unary_union(line_geometries)
        polygons = list(polygonize(linework))
        print(f"Created {len(polygons)} raw minimal polygons")
        if not polygons:
            raise ValueError("Road linework did not form any closed region polygons")

        sites_gdf = gpd.GeoDataFrame({"geometry": polygons}, geometry="geometry", crs="EPSG:3857")
        sites_gdf["area_m2"] = sites_gdf.geometry.area
        sites_gdf = sites_gdf[
            sites_gdf.geometry.is_valid
            & ~sites_gdf.geometry.is_empty
            & (sites_gdf["area_m2"] > self.min_area)
        ].copy()
        if sites_gdf.empty:
            raise ValueError("No valid region polygons remained after metric validation")

        sites_gdf["id"] = range(len(sites_gdf))
        sites = {"geometry": sites_gdf.geometry.tolist()}
        sites = pd.DataFrame(sites)
        sites["id"] = list(range(len(sites)))
        print(f"After metric validation: {len(sites)} sites")

        #----------FIX POLYGONS----------
        print("----------FIX POLYGONS----------")

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
        sites["geometry"] = sites.progress_apply(lambda row: simplify_points(row["geometry"]), axis=1)
        sites = sites[sites.progress_apply(lambda row: row["geometry"] != None, axis=1)]

        def collections_to_polygons(geometry):
            polygons = []
            for geom in geometry.geoms:
                geom = shapely.make_valid(geom, method='structure', keep_collapsed=False)
                if type(geom) == MultiPolygon:
                    polygons.extend(geom.geoms)
                elif type(geom) == GeometryCollection:
                    polygons.extend(collections_to_polygons(geom))
                elif type(geom) == Polygon:
                    polygons.append(geom)
            return polygons

        def get_polygons(row):
            geometry = shapely.make_valid(row["geometry"], method='structure', keep_collapsed=False)
            if type(geometry) in [MultiPolygon, GeometryCollection]:
                polygons = collections_to_polygons(geometry)
                polygons = [polygon for polygon in polygons if type(polygon) == Polygon]
                if len(polygons) == 0:
                    return None
                return polygons
            elif type(geometry) == Polygon:
                return [geometry]
            else:
                return None
        sites["geometry"] = sites.progress_apply(lambda row: get_polygons(row), axis=1)
        sites = sites[sites.progress_apply(lambda row: row["geometry"] != None, axis=1)]

        regions_dict = {"id":[], "geometry":[]}
        def get_regions(row):
            polygons = row["geometry"]
            for polygon in polygons:
                if not polygon.is_valid:
                    try:
                        polygon = shapely.make_valid(polygon, method='structure', keep_collapsed=False)
                    except:
                        continue
                regions_dict["id"].append(len(regions_dict["id"]))
                regions_dict["geometry"].append(polygon)
        sites.progress_apply(lambda row: get_regions(row), axis=1)
        sites = pd.DataFrame(regions_dict)

        def is_thin_or_irregular(polygon):
            circularity = 4 * math.pi * polygon.area / (polygon.length ** 2) if polygon.length > 0 else 0
            return circularity < self.min_circularity
        sites = sites[sites.progress_apply(lambda row: not is_thin_or_irregular(row["geometry"]), axis=1)]

        sites["region"] = sites.progress_apply(lambda row: self.shapely_to_polygons_converter(polygons=row["geometry"]), axis=1)
        sites = sites[sites.progress_apply(lambda row: len(row["region"]) > 0, axis=1)]

        def round_points(row):
            region = row["region"]
            n_digits = int(abs(math.log10(self.user_eps)))
            new_region = []
            for polygon in region:
                new_polygon = [(round(p[0], n_digits), round(p[1], n_digits)) for p in polygon]
                new_region.append(new_polygon)
            return new_region
        sites["region"] = sites.progress_apply(lambda row: round_points(row), axis=1)

        def is_valid(row):
            region = row["region"]
            region = [[(p[0], p[1]) for p in polygon] for polygon in region]
            try:
                region = self.polygons_to_shapely_converter(polygons=region)
                return region.is_valid
            except:
                return False
        sites["is_valid"] = sites.progress_apply(lambda row: is_valid(row), axis=1)
        sites = sites[sites["is_valid"]]
        
        site_dataset = sites[["id", "region"]].to_dict("list")

        return site_dataset
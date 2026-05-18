from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Tuple

import trimesh
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .core.base import Function

class MassingToTrimeshConverter(Function):
    def __init__(self) -> None:
        def polygons_to_shapely(polygons: List[List[Tuple[float, float]]]):
            shapely_cur_polygon = Polygon(polygons[0])
            for i in range(1, len(polygons)):
                poly = Polygon(polygons[i])
                if shapely_cur_polygon.equals(poly):
                    continue
                elif shapely_cur_polygon.contains_properly(poly):
                    shapely_cur_polygon = shapely_cur_polygon.difference(poly)
                elif poly.contains_properly(shapely_cur_polygon):
                    shapely_cur_polygon = poly.difference(shapely_cur_polygon)
                else:
                    shapely_cur_polygon = shapely_cur_polygon.union(poly)
            return shapely_cur_polygon
        self.polygons_to_shapely = polygons_to_shapely
        def extrude_shapely_polygon(polygon: Polygon, base_elev: float, height: float) -> trimesh.Trimesh:
            exterior_coords = list(polygon.exterior.coords)
            if exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            holes_coords = []
            for interior in polygon.interiors:
                interior_coords = list(interior.coords)
                if interior_coords[0] == interior_coords[-1]:
                    interior_coords = interior_coords[:-1]
                holes_coords.append(interior_coords)
            exterior_array = np.array(exterior_coords)
            holes_arrays = [np.array(hole) for hole in holes_coords]
            poly_2d = trimesh.path.polygons.Polygon(exterior_array, holes=holes_arrays)
            mesh = trimesh.creation.extrude_polygon(poly_2d, height=height, engine="earcut")
            mesh.apply_translation([0, 0, base_elev])
            return mesh
        self.extrude_shapely_polygon = extrude_shapely_polygon

    def __call__(self, *, massing: List[Dict[str, Any]]) -> trimesh.Trimesh:
        meshes = []
        for extrusion in massing:
            bottom_elevation = extrusion["bottom_elevation"]
            top_elevation = extrusion["top_elevation"]
            height = top_elevation - bottom_elevation
            polygons = extrusion["polygons"]
            
            shapely_geom = self.polygons_to_shapely(polygons)
            layer_meshes = []
            if isinstance(shapely_geom, MultiPolygon):
                for poly in shapely_geom.geoms:
                    mesh = self.extrude_shapely_polygon(poly, bottom_elevation, height)
                    layer_meshes.append(mesh)
            elif isinstance(shapely_geom, Polygon):
                mesh = self.extrude_shapely_polygon(shapely_geom, bottom_elevation, height)
                layer_meshes.append(mesh)
            layer_mesh = None
            if layer_meshes:
                result = layer_meshes[0].copy()
                for i in range(1, len(layer_meshes)):
                    try:
                        result = result.union(layer_meshes[i])
                    except Exception as e:
                        result = trimesh.util.concatenate([result, layer_meshes[i]])
                layer_mesh = result
            meshes.append(layer_mesh)
        
        combined_mesh = meshes[0].copy()
        for i in range(1, len(meshes)):
            try:
                combined_mesh = combined_mesh.union(meshes[i])
            except Exception as e:
                combined_mesh = trimesh.util.concatenate([combined_mesh, meshes[i]])
        
        return combined_mesh
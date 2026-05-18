from typing import List, Dict, Any
import re

import trimesh
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .mesh_compiler import MeshCompiler
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter

class PolygonMeshCompiler(MeshCompiler):
    def __init__(self) -> None:
        self.polygons_converter = PolygonsToShapelyConverter()
        def get_mesh(site):
            exterior_coords = list(site.exterior.coords)
            if exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            holes_coords = []
            for interior in site.interiors:
                interior_coords = list(interior.coords)
                if interior_coords[0] == interior_coords[-1]:
                    interior_coords = interior_coords[:-1]
                holes_coords.append(interior_coords)
            exterior_array = np.array(exterior_coords)
            holes_arrays = [np.array(hole) for hole in holes_coords]
            poly_2d = trimesh.path.polygons.Polygon(exterior_array, holes=holes_arrays)
            site_mesh = trimesh.creation.extrude_polygon(poly_2d, height=0.5)
            return site_mesh
        self.get_mesh = get_mesh

    def __call__(self, *, obj: Any) -> trimesh.Trimesh:
        site = [[(p[0], p[1]) for p in polygon] for polygon in obj]
        site = self.polygons_converter(polygons=site)
        meshes = []
        if isinstance(site, MultiPolygon):
            for poly in site.geoms:
                local_mesh = self.get_mesh(poly)
                meshes.append(local_mesh)
        elif isinstance(site, Polygon):
            local_mesh = self.get_mesh(site)
            meshes.append(local_mesh) 
        site_mesh = trimesh.util.concatenate(meshes)
        return site_mesh  


        
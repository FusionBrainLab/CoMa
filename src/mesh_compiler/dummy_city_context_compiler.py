from typing import List, Dict, Any
import re

import trimesh
import numpy as np
import shapely.geometry as sg

from .mesh_compiler import MeshCompiler
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..polys_geo_projector import PolysGeoProjector

class DummyCityContextCompiler(MeshCompiler):
    def __init__(self) -> None:
        self.polygons_converter = PolygonsToShapelyConverter()
        self.geo_projector = PolysGeoProjector()

    def __call__(self, *, obj: Any) -> trimesh.Trimesh:
        site = self.geo_projector(polys=obj)
        base_point = site[0][0]
        site = [[(p[0] - base_point[0], p[1] - base_point[1]) for p in polygon] for polygon in site]
        site = self.polygons_converter(polygons=site)

        minx, miny, maxx, maxy = site.bounds

        shift = 20
        size = 30
        rects = [
            [[minx-shift, miny-shift], [minx-shift-size, miny-shift], [minx-shift-size, miny-shift-size], [minx-shift, miny-shift-size]],
            [[minx-shift, maxy+shift], [minx-shift-size, maxy+shift], [minx-shift-size, maxy+shift+size], [minx-shift, maxy+shift+size]],
            [[maxx+shift, maxy+shift], [maxx+shift+size, maxy+shift], [maxx+shift+size, maxy+shift+size], [maxx+shift, maxy+shift+size]],
            [[maxx+shift, miny-shift], [maxx+shift+size, miny-shift], [maxx+shift+size, miny-shift-size], [maxx+shift, miny-shift-size]]
        ]

        meshes = []
        for rect_coords in rects:
            rect_poly = sg.Polygon(rect_coords)
            mesh = trimesh.creation.extrude_polygon(rect_poly, height=20)
            meshes.append(mesh)
        final_mesh = trimesh.util.concatenate(meshes)
        return final_mesh
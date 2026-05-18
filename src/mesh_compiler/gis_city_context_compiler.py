from typing import List, Dict, Any
import re

import trimesh
import numpy as np
import shapely.geometry as sg
import requests
import shapely
from tqdm import tqdm

from .mesh_compiler import MeshCompiler
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..polys_geo_projector import PolysGeoProjector
from .massing_mesh_compiler import MassingMeshCompiler
from ..point_geo_projector import PointGeoProjector
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter

class GISCityContextCompiler(MeshCompiler):
    def __init__(self, *, url: str,
                        ca_cert_path: str,
                        client_cert_path: str,
                        radius: int) -> None:
        self.url = url
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.radius = radius
        self.floor_height = 3
        self.massing_compiler = MassingMeshCompiler()
        self.geo_projector = PolysGeoProjector()
        self.point_projector = PointGeoProjector()
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()

    def __call__(self, *, obj: Any) -> trimesh.Trimesh:
        print("Request gis")
        shapely_site = self.polygons_to_shapely_converter(polygons=obj)
        """base_point = self.point_projector(point=(obj[0][0][0], obj[0][0][1]))
        context_data = []
        for polygon in obj:
            for point in tqdm(polygon):
                data = {
                    "lon":point[0],
                    "lat":point[1],
                    "radius":self.radius
                }
                print(data)
                response = requests.get(self.url, params=data, verify=self.ca_cert_path, cert=self.client_cert_path)
                output_data = response.json()
                print(output_data)
                context_data.append(output_data)
        print("Total output data")
        print(context_data)"""

        base_point = shapely.centroid(shapely_site)
        base_point = (base_point.x, base_point.y)
        convex_site = shapely.convex_hull(shapely_site)
        convex_site = self.polygons_to_shapely_converter(polygons=self.geo_projector(polys=self.shapely_to_polygons_converter(polygons=convex_site)))
        add_radius = shapely.minimum_bounding_radius(convex_site)

        context_data = []
        data = {
            "lon":base_point[0],
            "lat":base_point[1],
            "radius":self.radius + int(add_radius)
        }
        print(data)
        response = requests.get(self.url, params=data, verify=self.ca_cert_path, cert=self.client_cert_path)
        output_data = response.json()
        print(len(output_data["items"]))
        context_data.append(output_data)
        print("Total output data")
        print(context_data)

        base_point = self.point_projector(point=(obj[0][0][0], obj[0][0][1]))
        massings = []
        last_floors_count = 2
        shapely_context = None
        for output_data in context_data:
            for item in output_data["items"]:
                polygons = item["geometryHover"]
                polygons = self.shapely_to_polygons_converter(polygons=shapely.from_wkt(polygons))
                shapely_polygons = self.polygons_to_shapely_converter(polygons=polygons)

                if shapely.area(shapely_site.intersection(shapely_polygons)) > 0:
                    continue
                if shapely_context != None:
                    if shapely.area(shapely_context.intersection(shapely_polygons)) > 0:
                        continue

                if shapely_context == None:
                    shapely_context = shapely_polygons
                else:
                    shapely_context = shapely_context.union(shapely_polygons)

                polygons = self.geo_projector(polys=polygons)
                polygons = [[(p[0] - base_point[0], p[1] - base_point[1]) for p in polygon] for polygon in polygons]
                floors_count = item["floorsCount"] if item["floorsCount"] != None else last_floors_count
                last_floors_count = floors_count
                height = floors_count * self.floor_height
                massing = [
                    {
                        "id":0,
                        "massing":[
                            {
                                "polygons":polygons,
                                "bottom_elevation":0,
                                "top_elevation":height
                            }
                        ]
                    }
                ]
                massings.append(massing)
        meshes = [self.massing_compiler(obj=m) for m in massings]
        mesh = trimesh.util.concatenate(meshes)
        return mesh

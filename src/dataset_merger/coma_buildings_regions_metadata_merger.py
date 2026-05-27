from typing import Any, Dict, List

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from tqdm import tqdm

from .dataset_merger import DatasetMerger
from ..dataset_creator import CsvDatasetCreator
from ..polygons_to_shapely_converter import PolygonsToShapelyConverter
from ..shapely_to_polygons_converter import ShapelyToPolygonsConverter
from ..dataset_merger import GeoPandasDatasetIntersector, GeoPandasRadiusDatasetJoiner

class CoMaBuildingsRegionsMetadataMerger(DatasetMerger):
    def __init__(self, *, buildings_dataset_key: str,
                        regions_dataset_key: str,
                        buildings_col: str,
                        regions_col: str,
                        metadata_path: str,
                        metadata_property_col: str,
                        metadata_time_col: str,
                        metadata_feature_cols: List[str],
                        metadata_features_renaming: Dict[str, str],
                        footprint_region_geo_match_threshold: float,
                        features_col: str,
                        massing_col: str,
                        context_radiuses: List[int]) -> None:
        self.buildings_dataset_key = buildings_dataset_key
        self.regions_dataset_key = regions_dataset_key
        self.buildings_col = buildings_col
        self.regions_col = regions_col
        self.metadata_path = metadata_path
        self.metadata_property_col = metadata_property_col
        self.metadata_time_col = metadata_time_col
        self.metadata_feature_cols = metadata_feature_cols
        self.metadata_features_renaming = metadata_features_renaming
        self.footprint_region_geo_match_threshold = footprint_region_geo_match_threshold
        self.polygons_to_shapely_converter = PolygonsToShapelyConverter()
        self.features_col = features_col
        self.massing_col = massing_col
        self.context_radiuses = context_radiuses
        self.shapely_to_polygons_converter = ShapelyToPolygonsConverter()

    def __call__(self, *, datasets: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        tqdm.pandas()
        #----------LOAD DATASETS----------
        buildings_dataset = datasets[self.buildings_dataset_key]
        regions_dataset = datasets[self.regions_dataset_key]
        
        pd_buildings = pd.DataFrame(buildings_dataset)
        pd_buildings["id_buffer"] = pd_buildings["id"]
        pd_buildings = pd_buildings.set_index("id_buffer")
        pd_regions = pd.DataFrame(regions_dataset)
        pd_regions["id_buffer"] = pd_regions["id"]
        pd_regions = pd_regions.set_index("id_buffer")
        
        metadata_loader = CsvDatasetCreator(
            path=self.metadata_path,
            pandas_args={
                "sep":",",
                "index_col":False
            }
        )
        metadata_dataset = metadata_loader()
        pd_metadata = pd.DataFrame(metadata_dataset)

        #----------MERGE BUILDINGS AND REGIONS----------

        def get_contour(row):
            polygons = [[(p[0], p[1]) for p in polygon] for polygon in row[self.regions_col]]
            contour = self.polygons_to_shapely_converter(polygons=polygons)
            return contour
        pd_regions[self.regions_col] = pd_regions.progress_apply(lambda row: get_contour(row), axis=1)

        def get_footprint(row):
            shapely_polygons = []
            for e in row[self.buildings_col]:
                polygons = [[(p[0], p[1]) for p in polygon] for polygon in e["polygons"]]
                shapely_polygons.append(self.polygons_to_shapely_converter(polygons=polygons))
            footprint = unary_union(shapely_polygons)
            return footprint
        pd_buildings["footprint"] = pd_buildings.progress_apply(lambda row: get_footprint(row), axis=1)

        intersector = GeoPandasDatasetIntersector(
            intersect_dataset_key="intersect_dataset",
            intersect_dataset_geo_col="footprint",
            intersect_dataset_id_col="id",
            main_dataset_key="main_dataset",
            main_dataset_geo_col=self.regions_col,
            main_dataset_id_col="id",
            drop_self_intersections=False,
            output_properties=["id", "area"],
            output_col="intersections"
        )
        datasets = {
            "intersect_dataset": pd_buildings.to_dict("list"),
            "main_dataset": pd_regions.to_dict("list")
        }
        regions = intersector(datasets=datasets)
        pd_regions = pd.DataFrame(regions)

        building_areas = {row["id"]:row["footprint"].area for _, row in pd_buildings.iterrows()}
        pd_regions["building_ids"] = pd_regions.progress_apply(lambda row: [spec["id"] for spec in row["intersections"] if spec["area"]/building_areas[spec["id"]] > self.footprint_region_geo_match_threshold], axis=1)

        regions_to_buildings = {}
        buildings_to_regions = {}
        for _, row in pd_regions.iterrows():
            for building_id in row["building_ids"]:
                if building_id not in buildings_to_regions:
                    buildings_to_regions[building_id] = []
                buildings_to_regions[building_id].append(row["id"])
            regions_to_buildings[row["id"]] = row["building_ids"]

        #----------MERGE BUILDINGS AND METADATA----------

        regions_to_properties = {row["id"]:row["properties"] for _, row in pd_regions.iterrows()}
        pd_buildings = pd_buildings[pd_buildings.progress_apply(lambda row: row["id"] in buildings_to_regions, axis=1)]

        def get_features(row):
            features = {n: None for n in self.metadata_features_renaming.values()}
            local_regions = buildings_to_regions[row["id"]]
            local_properties = list(set([prop for region in local_regions for prop in regions_to_properties[region]]))
            local_metadata = pd_metadata[pd_metadata[self.metadata_property_col].isin(local_properties)]
            if len(local_metadata) == 0:
                return features
            max_time = max(local_metadata[self.metadata_time_col].values.tolist())
            local_metadata = local_metadata[local_metadata[self.metadata_time_col] == max_time]

            if len(local_metadata) == 0:
                return features

            for feature in self.metadata_feature_cols:
                local_values = local_metadata[local_metadata[feature].notna()][feature].values.tolist()
                if len(local_values) == 0:
                    continue
                features[self.metadata_features_renaming[feature]] = local_values[0]
            return features
        pd_buildings[self.features_col] = pd_buildings.progress_apply(lambda row: get_features(row), axis=1)

        #----------GET MASSINGS----------

        massings_dict = {"id":[], self.massing_col:[], self.regions_col:[], self.features_col:[]}
        def get_massings(row):
            buildings = pd_buildings.loc[row["building_ids"]]
            massing = [{"id":str(i), "massing": b[1][self.buildings_col]} for i, b in enumerate(buildings.iterrows())]
            features = [{"id":str(i), **b[1][self.features_col]} for i, b in enumerate(buildings.iterrows())]

            massings_dict["id"].append(str(len(massings_dict["id"])))
            massings_dict[self.massing_col].append(massing)
            massings_dict[self.regions_col].append(row[self.regions_col])
            massings_dict[self.features_col].append(features)
        pd_regions.progress_apply(lambda row: get_massings(row), axis=1)

        pd_massings = pd.DataFrame(massings_dict)
        pd_massings = pd_massings[pd_massings.progress_apply(lambda row: len(row[self.massing_col]) > 0, axis=1)]
        
        #----------GET CONTEXT----------

        for context_radius in tqdm(self.context_radiuses):
            context_joiner = GeoPandasRadiusDatasetJoiner(
                main_dataset_key="main_dataset",
                join_dataset_key="join_dataset",
                geo_col=self.regions_col,
                radius=context_radius,
                id_col="id",
                joined_ids_col=f"context_ids_{context_radius}"
            )
            datasets = {
                "main_dataset": pd_massings.to_dict("list"),
                "join_dataset": pd_massings.to_dict("list")
            }
            context = context_joiner(datasets=datasets)
            pd_massings = pd.DataFrame(context)
            pd_massings[f"context_ids_{context_radius}"] = pd_massings.progress_apply(lambda row: [i for i in row[f"context_ids_{context_radius}"] if i != row["id"]], axis=1)

        pd_massings[self.regions_col] = pd_massings.progress_apply(lambda row: self.shapely_to_polygons_converter(polygons=row[self.regions_col]), axis=1)

        pd_massings["base_point"] = pd_massings.progress_apply(lambda row: row[self.regions_col][0][0], axis=1)

        def get_relative_massing(row):
            base_point = row["base_point"]
            new_massing = []
            for m in row[self.massing_col]:
                new_m = {"id":m["id"], "massing":[]}
                for e in m["massing"]:
                    new_e = {"polygons":[]}
                    for polygon in e["polygons"]:
                        new_p = [(round(p[0]-base_point[0], 2), round(p[1]-base_point[1], 2)) for p in polygon]
                        new_e["polygons"].append(new_p)
                    new_e["bottom_elevation"] = e["bottom_elevation"]
                    new_e["top_elevation"] = e["top_elevation"]
                    new_m["massing"].append(new_e)
                new_massing.append(new_m)
            return new_massing
        pd_massings[self.massing_col] = pd_massings.progress_apply(lambda row: get_relative_massing(row), axis=1)
        
        def get_relative_site(row):
            base_point = row["base_point"]
            new_site = []
            for polygon in row[self.regions_col]:
                new_p = [(round(p[0]-base_point[0], 2), round(p[1]-base_point[1], 2)) for p in polygon]
                new_site.append(new_p)
            return new_site
        pd_massings[self.regions_col] = pd_massings.progress_apply(lambda row: get_relative_site(row), axis=1)
        
        return pd_massings.to_dict("list")
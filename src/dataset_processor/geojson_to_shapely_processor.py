from typing import List, Any, Dict

from tqdm import tqdm
import shapely
import pandas as pd

from .dataset_processor import DatasetProcessor

class GeoJSONToShapelyProcessor(DatasetProcessor):
    def __init__(self, *, geojson_col: str,
                        on_invalid: str,
                        output_col: str) -> None:
        self.geojson_col = geojson_col
        self.on_invalid = on_invalid
        self.output_col = output_col
        
    def __call__(self, *, dataset: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        tqdm.pandas()

        pd_dataset = pd.DataFrame(dataset)
        pd_dataset[self.output_col] = pd_dataset.progress_apply(lambda row: shapely.from_geojson(row[self.geojson_col], on_invalid=self.on_invalid), axis=1)

        output_dataset = pd_dataset.to_dict("list")
        return output_dataset

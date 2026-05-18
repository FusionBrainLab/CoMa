# Dataset Postprocessing and Integration

## Overview
This experiment performs final dataset postprocessing by converting massing geometries to local coordinate systems and integrating all generated image assets into the dataset structure.

## Operations Performed
- **Coordinate system conversion**: Transforms massing geometries from global to local coordinate systems for consistency
- **Image path integration**: Adds columns with file paths to all generated images from previous experiments:
  - Images from `data_processing/231025_images_sampling` (environment and map views)
  - Renderings from `data_processing/261025_images_rendering` (low and high-quality environment renders)

## Input Data
- Dataset from previous processing stages
- Image assets from:
  - `data_processing/231025_images_sampling`
  - `data_processing/261025_images_rendering`

## Output
- Finalized `dataset.json` with local coordinates and integrated image paths

## Execution
```bash
python main.py
```
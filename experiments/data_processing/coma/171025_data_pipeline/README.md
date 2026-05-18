# Massing Creation Pipeline

## Overview
This experiment implements the core "Massing Creation" stage of the data pipeline as defined in the CoMa paper. It processes polygon datasets and metadata to create building massings with associated functional-economical requirements.

## Operations Performed
- **Top-level sites filtering**: Filters property boundaries to retain only top-level, physically distinct development sites
- **Buildings creation**: Merges individual footprint extrusions into complete building massings
- **Sites-to-building matching**: Matches buildings to development sites using spatial intersection (>90% area threshold)
- **Metadata-to-buildings merging**: Combines building information, residential dwellings, business establishments, and public spaces data
- **Floor height and area computing**: Calculates mean floor height and total usable area from geometry
- **Meshes creation**: Compiles massing geometry into standard 3D mesh format
- **Buildings-to-sites merging**: Groups finalized buildings back to their associated sites

## Operations Not Included
- Public massings filtering
- Sites combining

## Input Data
- Polygon datasets from intersection computation experiment (data_processing/171025_total_intersection_resaving)
- Metadata tables (Building Information, Residential Dwellings, Business Establishments, Public Spaces)

## Output
- `dataset.json` file with prepared massing dataset

## Execution
```bash
python main.py
```

## Notes
- Implements the core massing creation workflow described in Section 3.3 of the CoMa paper
- Results form the foundation for subsequent environment creation and filtering stages
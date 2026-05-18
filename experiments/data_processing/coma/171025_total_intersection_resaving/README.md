# Dataset Intersection Computation

## Overview
This experiment computes all-to-all intersection areas between three polygon datasets:
- Property Boundaries
- Building Footprints 
- CLUE Blocks

The computation generates modified versions of all three datasets with additional columns containing intersection area information.

## Input Datasets
- Property Boundaries dataset
- Building Footprints dataset  
- CLUE Blocks dataset

## Output
Three modified polygon datasets with additional intersection area columns:
- Property Boundaries with intersection areas
- Building Footprints with intersection areas
- CLUE Blocks with intersection areas

## Execution
```bash
python main.py
```

## Notes
- Computes geometric intersections between all polygon combinations
- Adds intersection area metadata to original datasets
- Results are used for subsequent spatial analysis and dataset linking
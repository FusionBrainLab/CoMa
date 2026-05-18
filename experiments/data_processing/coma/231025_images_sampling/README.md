# Environment Visualization Generation

## Overview
This experiment creates environmental context for each development site by generating surrounding environment meshes and sampling contextual images.

## Operations Performed
- **Environment meshes creation**: Constructs unified 3D environment meshes by combining buildings from environment sites
- **Environment images sampling**: Generates multiple views of environment meshes with target site contours overlaid in red

## Input Data
- `dataset.json` from `data_processing/241025_data_pipeline` (combined sites)
- `dataset.json` from `data_processing/171025_data_pipeline` (base sites)

## Output
- Environment meshes for each development site in subfolder env_mesh
- Contextual images saved in organized subfolders:
  - env_image
  - map_image_alt
  - map_image_base

## Execution
```bash
python main.py
```

## Notes
- Implements the initial environment creation steps from Section 3.3 of the CoMa paper
- Camera poses are optimized to maximize target site visibility
- Forms the foundation for subsequent rendering enhancement stages
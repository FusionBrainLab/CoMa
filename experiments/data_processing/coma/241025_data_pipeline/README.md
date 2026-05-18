# Massing Filtering and Site Combination

## Overview
This experiment implements the final two operations of the "Massing Creation" pipeline stage: public massings filtering and sites combining, completing the massing dataset preparation.

## Operations Performed
- **Public massings filtering**: Filters out private houses and townhouses, retaining only public-facing properties such as apartment complexes and commercial offices
- **Sites combining**: Programmatically combines 2 to 4 neighboring properties into larger, aggregated development sites to increase dataset diversity

## Input Data
- `dataset.json` from the previous massing creation experiment (data_processing/171025_data_pipeline)

## Output
- Updated `dataset.json` file with filtered and combined massing dataset

## Execution
```bash
python main.py
```
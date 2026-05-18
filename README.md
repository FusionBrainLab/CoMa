# Massing Generation Research Repository

A comprehensive research framework for building massing generation using computer vision and deep learning techniques.

## Repository Structure

```
.
├── data/                    # Unstructured data common to all experiments
├── experiments/             # Individual experiments and pipeline steps
├── papers/                  # Research papers and publications
├── sandbox/                 # Temporary scripts and experiments
├── src/                     # Main source code
└── run.py                  # Main entry point for running experiments
```

- **experiments/**: Contains experiments with nested organization, where each leaf folder represents a distinct experiment or pipeline step with description, results, and corresponding thoughts. Experiments can be executed with Python scripts inside or via JSON config files using the root `run.py` script.

## Code Architecture

This section describes how to organize code in the `src/` folder:

- **One class per file** principle
- **Independent classes** (without abstract parents) are placed directly in `src/` root as `.py` files
- **Abstract classes with implementations**: Each abstract class is organized in its own subfolder where:
  - One file contains the abstract class (interface)
  - Other files in the same folder contain inherited concrete classes
- **Configuration-driven execution** via JSON config files using `run.py`

### Running Experiments

```bash
# Default execution command
python run.py --config_path path/to/config.json
```

*Note: Always check the experiment's README for specific instructions, dependencies, and execution commands as some experiments may require different execution methods (e.g., torchrun for distributed training).*

---

## Research Roadmap

### 📄 CoMa: Contextual Massing Generation with Vision-Language Models

**Paper Location:** `papers/coma/`

#### Problem Statement
Automating the architectural massing design phase, which is traditionally manual and relies heavily on designer intuition, by creating a data-driven approach that integrates functional requirements with visual urban context.

#### Key Idea
Formulate massing generation as a conditional vision-language task where models generate 3D building geometries based on textual requirements and visual context of the development site.

#### Main Contributions
1. **CoMa-20K Dataset**: First large-scale multi-modal dataset pairing massing geometries with functional requirements and urban context views
2. **VLM Benchmark**: Comprehensive evaluation of both fine-tuned and zero-shot vision-language models
3. **Contextual Evaluation**: Novel metrics for assessing structural accuracy and contextual relevance

#### Results
- Demonstrated feasibility of VLM-based massing generation
- Revealed trade-offs: fine-tuned models attempt complex geometries but suffer artifacts; zero-shot models produce clean but simplistic forms
- Established foundational benchmark for future research

#### Implementation Details

**Dataset Location:** `experiments/data_processing/181125_dataset_resaving/`

#### Reproduction Pipeline

To reproduce the CoMa paper results, discover and explore the following experiments in order. Each experiment contains detailed instructions and descriptions in its respective README:

1. `data_processing/171025_total_intersection_resaving`
2. `data_processing/171025_data_pipeline`
3. `data_processing/241025_data_pipeline`
4. `data_processing/231025_images_sampling`
5. `data_processing/261025_images_rendering`
6. `data_processing/271025_final_data_processing`
7. `data_processing/071125_train_test_split`
8. `data_processing/091125_dataset_overview`
9. `training/071125_system_prompt_train`
10. `training/121125_zero_shot_inference`

Refer to each experiment's README for specific running instructions, requirements, and additional details about the pipeline steps.
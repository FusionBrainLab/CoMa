HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_1d_3d_visualization.yaml \
--multirun model_size=2,4,8 context_2d_count=0,1 x_feature=context_1d_count line_feature=context_3d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_1d_3d_mean_visualization.yaml \
x_feature=context_1d_count line_feature=context_3d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_1d_3d_visualization.yaml \
--multirun model_size=2,4,8 context_2d_count=0,1 x_feature=context_3d_count line_feature=context_1d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_1d_3d_mean_visualization.yaml \
x_feature=context_3d_count line_feature=context_1d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_1d_visualization.yaml \
--multirun context_2d_count=0,1 context_3d_count=0,2,4,6,8 x_feature=model_size line_feature=context_1d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_1d_mean_visualization.yaml \
x_feature=model_size line_feature=context_1d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_1d_visualization.yaml \
--multirun context_2d_count=0,1 context_3d_count=0,2,4,6,8 x_feature=context_1d_count line_feature=model_size

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_1d_mean_visualization.yaml \
x_feature=context_1d_count line_feature=model_size

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_3d_visualization.yaml \
--multirun context_2d_count=0,1 context_1d_count=0,2,4,6,8,10 x_feature=model_size line_feature=context_3d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_3d_mean_visualization.yaml \
x_feature=model_size line_feature=context_3d_count

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_3d_visualization.yaml \
--multirun context_2d_count=0,1 context_1d_count=0,2,4,6,8,10 x_feature=context_3d_count line_feature=model_size

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line/multimodal_line_model_3d_mean_visualization.yaml \
x_feature=context_3d_count line_feature=model_size


python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/unimodal_line_visualization.yaml
x_feature=context_1d_count line_feature=model_size zero_feature_1=context_2d_count zero_feature_2=context_3d_count

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/unimodal_line_visualization.yaml
x_feature=context_2d_count line_feature=model_size zero_feature_1=context_1d_count zero_feature_2=context_3d_count

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/unimodal_line_visualization.yaml
x_feature=context_3d_count line_feature=model_size zero_feature_1=context_1d_count zero_feature_2=context_2d_count

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/unimodal_line_visualization.yaml
x_feature=model_size line_feature=context_1d_count zero_feature_1=context_2d_count zero_feature_2=context_3d_count

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/unimodal_line_visualization.yaml
x_feature=model_size line_feature=context_2d_count zero_feature_1=context_1d_count zero_feature_2=context_3d_count

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/unimodal_line_visualization.yaml
x_feature=model_size line_feature=context_3d_count zero_feature_1=context_1d_count zero_feature_2=context_2d_count
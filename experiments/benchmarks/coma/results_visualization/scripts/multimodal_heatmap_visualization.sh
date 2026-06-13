python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_heatmap/multimodal_heatmap_1d_3d_visualization.yaml
--multirun model_size=2,4,8 context_2d_count=0,1

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_heatmap/multimodal_heatmap_1d_3d_mean_visualization.yaml

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_heatmap/multimodal_heatmap_model_1d_visualization.yaml
--multirun context_2d_count=0,1 context_3d_count=0,2,4,6,8

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_heatmap/multimodal_heatmap_model_1d_mean_visualization.yaml

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_heatmap/multimodal_heatmap_model_3d_visualization.yaml
--multirun context_2d_count=0,1 context_1d_count=0,2,4,6,8,10

python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_heatmap/multimodal_heatmap_model_3d_mean_visualization.yaml
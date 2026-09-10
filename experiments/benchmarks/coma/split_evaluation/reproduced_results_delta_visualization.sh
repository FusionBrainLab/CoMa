HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation_coma/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation_coma/experiments/benchmarks/coma/split_evaluation/reproduced_results_delta_visualization.yaml \
validation_type=lower

HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation_coma/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation_coma/experiments/benchmarks/coma/split_evaluation/reproduced_results_delta_visualization.yaml \
validation_type=upper

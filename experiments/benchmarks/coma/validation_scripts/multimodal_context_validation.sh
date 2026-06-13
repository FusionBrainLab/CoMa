HYDRA_FULL_ERROR=1 python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_run.py \
--config_path /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/validation_config.yaml \
--multirun \
model_size=2,4,8 \
train_type=multimodal_context \
context_1d_count=0,2,4,6,8,10 \
context_2d_count=0,1 \
context_3d_count=0,2,4,6,8

import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client, extra_options = training_job_api_from_profile('default')

    workdir = os.getcwd()

    author_name = 'maslov'

    dims = {
        "context_1d_count": [0, 2, 4, 6, 8, 10],
        "context_2d_count": [0, 1],
        "context_3d_count": [0, 2, 4, 6, 8],
    }
    """dims = {
        "context_1d_count": [10],
        "context_2d_count": [1],
        "context_3d_count": [8],
    }"""

    runs = {"2b_4b": [], "8b": []}
    for key in runs.keys():
        runs[key].append({
            "train_type":"no_context", "context_1d_count": 0, "context_2d_count": 0, "context_3d_count": 0,
        })

        runs[key].append({
            "train_type": "unimodal_context",
            "context_1d_count": 0,
            "context_2d_count": 0,
            "context_3d_count": 0,
        })

        for dim, values in dims.items():
            zero_dims = {k: 0 for k in dims.keys() if k != dim}
            for value in values:
                if value == 0:
                    continue
                runs[key].append({
                    **zero_dims,
                    dim: value,
                    "train_type": "unimodal_context",
                })
        
        for v1 in dims["context_1d_count"]:
            for v2 in dims["context_2d_count"]:
                for v3 in dims["context_3d_count"]:
                    runs[key].append({
                        "train_type": "multimodal_context",
                        "context_1d_count": v1,
                        "context_2d_count": v2,
                        "context_3d_count": v3,
                    })
            
    commands = []
    descriptions = []
    for key in runs.keys():
        for run in runs[key]:
            base_command = "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/envs/coma_inference/bin/python /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/hydra_batch_run.py --base_config /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/methods_validation/configs/base.yaml"
            batch_config = f"--batch_config /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/maslov/massing_generation/experiments/benchmarks/coma/methods_validation/configs/batch_{key}.yaml"
            args = []
            models = key.split("_")
            for model in models:
                for name, value in run.items():
                    args.append(f"runs.qwen_{model}_0.{name}={value}")
            commands.append(" ".join([base_command, batch_config, *args]))
            descriptions.append(f'#coma_inference {key}' + " ".join(args))
    for i, data in enumerate(zip(commands, descriptions)):
        command, description = data
        result = client.run_job(
            payload={
                'script': command,
                'job_desc': description,
                'env_variables': {
                    'HF_HOME': "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/transformers_cache",
                },
                'instance_type': 'a100plus.1gpu.80vG.12C.182G',
                'region': extra_options['region'],
                'type': 'binary',
                'base_image': 'cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41',
                'n_workers': 1,              # Количество воркеров.
                'processes_per_worker': 1,   # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
            }
        )


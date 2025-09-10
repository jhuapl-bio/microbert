#!/bin/bash
cd /home/apluser/analysis/analysis/experiment


accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/nucleotide-transformer-v2-50m-multi-species.yaml 
accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/hyenadna-small-32k-seqlen-hf.yaml 
accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/hyenadna-medium-160k-seqlen-hf.yaml 
accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/nucleotide-transformer-v2-100m-multi-species.yaml
accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/hyenadna-large-1m-seqlen-hf.yaml 
accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/nucleotide-transformer-v2-250m-multi-species.yaml

accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/DNABERT-S.yaml 
accelerate launch --config_file /home/apluser/.cache/huggingface/accelerate/default_config-8.yaml train_model_multi_gpu.py amr/multi/DNABERT-2-117M.yaml 

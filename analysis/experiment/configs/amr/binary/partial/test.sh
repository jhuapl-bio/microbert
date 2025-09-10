#!/bin/bash
cd /home/apluser/analysis/analysis/experiment
python train_model_multi_gpu.py amr/binary/partial/DNABERT-2-117M.yaml 

python train_model_multi_gpu.py amr/binary/partial/DNABERT-S.yaml 

python train_model_multi_gpu.py amr/binary/partial/nucleotide-transformer-v2-100m-multi-species.yaml
python train_model_multi_gpu.py amr/binary/partial/nucleotide-transformer-v2-250m-multi-species.yaml
python train_model_multi_gpu.py amr/binary/partial/nucleotide-transformer-v2-50m-multi-species.yaml 


python train_model_multi_gpu.py amr/binary/partial/hyenadna-large-1m-seqlen-hf.yaml 
python train_model_multi_gpu.py amr/binary/partial/hyenadna-medium-160k-seqlen-hf.yaml 
python train_model_multi_gpu.py amr/binary/partial/hyenadna-small-32k-seqlen-hf.yaml 
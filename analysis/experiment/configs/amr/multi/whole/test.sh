#!/bin/bash
cd /home/apluser/analysis/analysis/experiment
python train_model_multi_gpu.py amr/multi/whole/DNABERT-2-117M.yaml 

python train_model_multi_gpu.py amr/multi/whole/DNABERT-S.yaml 

python train_model_multi_gpu.py amr/multi/whole/nucleotide-transformer-v2-100m-multi-species.yaml
python train_model_multi_gpu.py amr/multi/whole/nucleotide-transformer-v2-250m-multi-species.yaml
python train_model_multi_gpu.py amr/multi/whole/nucleotide-transformer-v2-50m-multi-species.yaml 


python train_model_multi_gpu.py amr/multi/whole/hyenadna-large-1m-seqlen-hf.yaml 
python train_model_multi_gpu.py amr/multi/whole/hyenadna-medium-160k-seqlen-hf.yaml 
python train_model_multi_gpu.py amr/multi/whole/hyenadna-small-32k-seqlen-hf.yaml 
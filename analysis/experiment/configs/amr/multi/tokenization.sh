#!/bin/bash
cd /home/apluser/analysis/analysis/experiment
# python train_data.py amr/multi/DNABERT-2-117M.yaml 

python train_data.py amr/multi/DNABERT-S.yaml 

# python train_data.py amr/multi/nucleotide-transformer-v2-100m-multi-species.yaml
# python train_data.py amr/multi/nucleotide-transformer-v2-250m-multi-species.yaml
python train_data.py amr/multi/nucleotide-transformer-v2-50m-multi-species.yaml 


# python train_data.py amr/multi/hyenadna-large-1m-seqlen-hf.yaml 
# python train_data.py amr/multi/hyenadna-medium-160k-seqlen-hf.yaml 
python train_data.py amr/multi/hyenadna-small-32k-seqlen-hf.yaml 
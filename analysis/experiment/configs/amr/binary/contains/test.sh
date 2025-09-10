#!/bin/bash
cd /home/apluser/analysis/analysis/experiment

# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "/home/apluser/analysis/analysis/process_amr/input/final_test/amr_with_100.tsv" "/home/apluser/analysis/analysis/process_amr/input/final_test/amr_with_100.tsv" 
# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "binary_contains_50" "binary_contains_50" 

# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "contains_50" "contains_50" 

# mkdir -p /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains_50
# find /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains -maxdepth 1 -type f -exec cp {} /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains_50/ \;


# python train_model_multi_gpu.py amr/binary/contains_50/DNABERT-2-117M.yaml 

# python train_model_multi_gpu.py amr/binary/contains_50/DNABERT-S.yaml 

# python train_model_multi_gpu.py amr/binary/contains_50/nucleotide-transformer-v2-100m-multi-species.yaml
# python train_model_multi_gpu.py amr/binary/contains_50/nucleotide-transformer-v2-250m-multi-species.yaml
# python train_model_multi_gpu.py amr/binary/contains_50/nucleotide-transformer-v2-50m-multi-species.yaml 


# python train_model_multi_gpu.py amr/binary/contains_50/hyenadna-large-1m-seqlen-hf.yaml 
# python train_model_multi_gpu.py amr/binary/contains_50/hyenadna-medium-160k-seqlen-hf.yaml 
# python train_model_multi_gpu.py amr/binary/contains_50/hyenadna-small-32k-seqlen-hf.yaml 


# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "/home/apluser/analysis/analysis/process_amr/input/final_test/amr_with_100.tsv" "/home/apluser/analysis/analysis/process_amr/input/final_test/amr_with_100.tsv" 
# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "binary_contains_50" "binary_contains_100" 

# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "contains_50" "contains_100" 

# mkdir -p /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains_100
# find /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains -maxdepth 1 -type f -exec cp {} /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains_100/ \;

# python train_model_multi_gpu.py amr/binary/contains_100/DNABERT-2-117M.yaml 

# python train_model_multi_gpu.py amr/binary/contains_100/DNABERT-S.yaml 

# python train_model_multi_gpu.py amr/binary/contains_100/nucleotide-transformer-v2-100m-multi-species.yaml
# python train_model_multi_gpu.py amr/binary/contains_100/nucleotide-transformer-v2-250m-multi-species.yaml
# python train_model_multi_gpu.py amr/binary/contains_100/nucleotide-transformer-v2-50m-multi-species.yaml 


# python train_model_multi_gpu.py amr/binary/contains_100/hyenadna-large-1m-seqlen-hf.yaml 
# python train_model_multi_gpu.py amr/binary/contains_100/hyenadna-medium-160k-seqlen-hf.yaml 
# python train_model_multi_gpu.py amr/binary/contains_100/hyenadna-small-32k-seqlen-hf.yaml 


# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "/home/apluser/analysis/analysis/process_amr/input/final_test/amr_with_100.tsv" "/home/apluser/analysis/analysis/process_amr/input/final_test/amr_with_random.tsv" --ext ".yaml"
# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "binary_contains_100" "binary_contains_random" --ext ".yaml"

# python /home/apluser/analysis/analysis/process_amr_multiclass/sub.py /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains "contains_100" "contains_random" --ext ".yaml"

# mkdir -p /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains_random
# find /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains -maxdepth 1 -type f -exec cp {} /home/apluser/analysis/analysis/experiment/configs/amr/binary/contains_random/ \;

python train_model_multi_gpu.py amr/binary/contains_random/DNABERT-2-117M.yaml 

python train_model_multi_gpu.py amr/binary/contains_random/DNABERT-S.yaml 

python train_model_multi_gpu.py amr/binary/contains_random/nucleotide-transformer-v2-100m-multi-species.yaml
python train_model_multi_gpu.py amr/binary/contains_random/nucleotide-transformer-v2-250m-multi-species.yaml
python train_model_multi_gpu.py amr/binary/contains_random/nucleotide-transformer-v2-50m-multi-species.yaml 


python train_model_multi_gpu.py amr/binary/contains_random/hyenadna-large-1m-seqlen-hf.yaml 
python train_model_multi_gpu.py amr/binary/contains_random/hyenadna-medium-160k-seqlen-hf.yaml 
python train_model_multi_gpu.py amr/binary/contains_random/hyenadna-small-32k-seqlen-hf.yaml 
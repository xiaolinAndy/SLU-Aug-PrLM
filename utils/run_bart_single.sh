#!/bin/bash

current_pth="YOUR_FOLDER_PATH/utils/"
ATIS_data="../data/atis/"
SNIPS_data="../data/snips/"
result_folder="YOUR_FOLDER_PATH/result/"
bart_folder="../bart"
gpu_id=0
seed=2020

# test for snips

# 1. finetune bart
# {327, 1308, 13084}
size=1308
cd ${bart_folder}
python run_delex_to_raw_single.py --train_data_file ${SNIPS_data}train_${size}_raw.txt --train_delex_file ${SNIPS_data}train_${size}_delex.txt --train_label_file ${SNIPS_data}train_${size}_label.txt --output ../result/snips/single_${size}/  --cache_dir cache/ --do_train --num_train_epochs 5 --logging_steps 1000 --save_steps 0 --overwrite_output_dir --overwrite_cache --seed ${seed} --cuda_id ${gpu_id} --model_name_or_path ../resources/bart_large

# 2. generate data
checkpoint=5
python get_delex_to_raw_single.py --gen_data_file ${SNIPS_data}train_${size}_raw.txt --gen_delex_file ${SNIPS_data}train_${size}_delex.txt --gen_label_file ${SNIPS_data}train_${size}_label.txt --output_file ../result/snips/single_${size}/checkpoint-${checkpoint}/gen_data.txt --output_delex_file ${SNIPS_data}train_${size}_single_delex.txt --output_label_file ${SNIPS_data}train_${size}_single_label.txt --model_type bart --model_name_or_path ../result/snips/single_${size}/checkpoint-${checkpoint}/ --seed ${seed} --cuda_id ${gpu_id}
python filter_single_data.py --delex_file ${SNIPS_data}train_${size}_single_delex.txt --raw_file ${SNIPS_data}train_${size}_raw.txt --label_file ${SNIPS_data}train_${size}_single_label.txt --gen_file ../result/snips/single_${size}/checkpoint-${checkpoint}/gen_data.txt --filter_file ../result/snips/single_${size}/checkpoint-${checkpoint}/filtered_data.txt --org_file ${SNIPS_data}train_${size}_

# test for atis

# 1. finetune bart
# {111, 447, 4478}
size=447
cd ${bart_folder}
python run_delex_to_raw_single.py --train_data_file ${ATIS_data}train_${size}_raw.txt --train_delex_file ${ATIS_data}train_${size}_delex.txt --train_label_file ${ATIS_data}train_${size}_label.txt --output ../result/atis/single_${size}/  --cache_dir cache/ --do_train --num_train_epochs 5 --logging_steps 100 --save_steps 0 --overwrite_output_dir --overwrite_cache --seed 2020 --cuda_id ${gpu_id} --model_name_or_path ../resources/bart_large

# 2. generate data
checkpoint=5
python get_delex_to_raw_single.py --gen_data_file ${ATIS_data}train_${size}_raw.txt --gen_label_file ${ATIS_data}train_${size}_label.txt --gen_delex_file ${ATIS_data}train_${size}_delex.txt --output_file ../result/atis/single_${size}/checkpoint-${checkpoint}/gen_data.txt --output_delex_file ${ATIS_data}train_${size}_single_delex.txt --output_label_file ${ATIS_data}train_${size}_single_label.txt --model_type bart --model_name_or_path ../result/atis/single_${size}/checkpoint-${checkpoint}/ --seed ${seed} --cuda_id ${gpu_id}
python filter_single_data.py --delex_file ${ATIS_data}train_${size}_single_delex.txt --raw_file ${ATIS_data}train_${size}_raw.txt --label_file ${ATIS_data}train_${size}_single_label.txt --gen_file ../result/atis/single_${size}/checkpoint-${checkpoint}/gen_data.txt --filter_file ../result/atis/single_${size}/checkpoint-${checkpoint}/filtered_data.txt --org_file ${ATIS_data}train_${size}_









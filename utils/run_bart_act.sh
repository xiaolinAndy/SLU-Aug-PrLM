#!/bin/bash

current_pth="YOUR_DIRECTORY_FOLDER/utils/"
ATIS_data="../data/atis/"
SNIPS_data="../data/snips/"
result_folder="YOUR_DIRECTORY_FOLDER/result/"
bart_folder="../bart"
gpu_id=0
seed=2020



# test for snips and size 327

# 1. finetune bart
# {327, 1308, 13084}
size=1308
cd ${bart_folder}
python run_act_to_raw.py --train_data_file ${SNIPS_data}train_${size}_raw.txt --train_intent_file ${SNIPS_data}train_${size}_intent.txt --train_label_file ${SNIPS_data}train_${size}_label.txt --output ../result/snips/act_${size}/  --cache_dir cache/ --do_train --num_train_epochs 5 --logging_steps 100 --save_steps 0 --overwrite_output_dir --overwrite_cache --seed ${seed} --cuda_id ${gpu_id} --model_name_or_path ../resources/bart_large/ --do_eval --eval_all_checkpoints --eval_data_file ${SNIPS_data}val_raw.txt --eval_intent_file ${SNIPS_data}val_intent.txt --eval_label_file ${SNIPS_data}val_label.txt

# 2. generate data
checkpoint=5
python get_act_to_raw.py --gen_data_file ${SNIPS_data}train_${size}_raw.txt --gen_label_file ${SNIPS_data}train_${size}_label.txt --gen_intent_file ${SNIPS_data}train_${size}_intent.txt --slot_dict_file ${SNIPS_data}train_${size}_slot_dict.json --output_file ../result/snips/act_${size}/checkpoint-${checkpoint}/gen_data.txt --output_act_file ../result/snips/act_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --model_type bart --model_name_or_path ../result/snips/act_${size}/checkpoint-${checkpoint}/ --seed ${seed} --cuda_id ${gpu_id} --change_input
python filter_act_data.py --raw_file ${SNIPS_data}train_${size}_raw.txt --delex_file ${SNIPS_data}train_${size}_delex.txt --act_file ../result/snips/act_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --gen_file ../result/snips/act_${size}/checkpoint-${checkpoint}/gen_data.txt --filter_file ../result/snips/act_${size}/checkpoint-${checkpoint}/filtered_data.txt --label_file ${SNIPS_data}train_${size}_label.txt


# test for atis
# 1. finetune bart
# {111, 447, 4478}
size=447
cd ${bart_folder}
python run_act_to_raw.py --train_data_file ${ATIS_data}train_${size}_raw.txt --train_intent_file ${ATIS_data}train_${size}_intent.txt --train_label_file ${ATIS_data}train_${size}_label.txt --output ../result/atis/act_int_${size}/  --cache_dir cache/ --do_train --num_train_epochs 5 --logging_steps 50 --save_steps 0 --overwrite_output_dir --overwrite_cache --seed ${seed} --cuda_id ${gpu_id} --model_name_or_path ../resources/bart_large/ --do_eval --eval_all_checkpoints --eval_data_file ${ATIS_data}val_raw.txt

#2. generate data
checkpoint=5
python get_act_to_raw.py --gen_data_file ${ATIS_data}train_${size}_raw.txt --gen_label_file ${ATIS_data}train_${size}_label.txt --gen_intent_file ${ATIS_data}train_${size}_intent.txt --slot_dict_file ${ATIS_data}train_${size}_slot_dict.json --output_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/gen_data.txt --output_act_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --model_type bart --model_name_or_path ../result/atis/act_int_${size}/checkpoint-${checkpoint}/ --seed ${seed} --cuda_id ${gpu_id} --change_input
python filter_act_data.py --raw_file ${ATIS_data}train_${size}_raw.txt --delex_file ${ATIS_data}train_${size}_delex.txt --act_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --gen_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/gen_data.txt --filter_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/filtered_data.txt --label_file ${ATIS_data}train_${size}_label.txt




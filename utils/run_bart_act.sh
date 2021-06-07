#!/bin/bash

current_pth="/sdc/htlin/Code/pretrain_slu_da/utils/"
ATIS_data="../data/atis/"
SNIPS_data="../data/snips/"
result_folder="/sdc/htlin/Code/pretrain_slu_da/result/"
bart_folder="../bart"
gpu_id=0
seed=2020

# test for atis
# 1. finetune bart
size=447
cd ${bart_folder}
#python run_act_to_raw.py --train_data_file ${ATIS_data}train_${size}_raw.txt --train_intent_file ${ATIS_data}train_${size}_intent.txt --train_label_file ${ATIS_data}train_${size}_label.txt --output ../result/atis/act_int_${size}/  --cache_dir cache/ --do_train --num_train_epochs 10 --logging_steps 50 --save_steps 0 --overwrite_output_dir --overwrite_cache --seed ${seed} --cuda_id ${gpu_id} --model_name_or_path ../../pretrained_model/bart_large/ --do_eval --eval_all_checkpoints --eval_data_file ${ATIS_data}val_raw.txt
#
##2. generate data
#for (( checkpoint = 1; checkpoint <= 5; checkpoint=checkpoint+1 ))
#do
#    python get_act_to_raw.py --gen_data_file ${ATIS_data}train_${size}_raw.txt --gen_label_file ${ATIS_data}train_${size}_label.txt --gen_intent_file ${ATIS_data}train_${size}_intent.txt --slot_dict_file ${ATIS_data}train_${size}_slot_dict.json --output_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/gen_data.txt --output_act_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --model_type bart --model_name_or_path ../result/atis/act_int_${size}/checkpoint-${checkpoint}/ --seed ${seed} --cuda_id ${gpu_id} --change_input
#    python filter_act_data.py --raw_file ${ATIS_data}train_${size}_raw.txt --delex_file ${ATIS_data}train_${size}_delex.txt --act_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --gen_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/gen_data.txt --filter_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/filtered_data.txt --label_file ${ATIS_data}train_${size}_label.txt
#done

#3. convert data
#cd ${current_pth}
#for (( checkpoint = 1; checkpoint <= 5; checkpoint=checkpoint+1 ))
#do
#    python preprocess/get_evaluate_data.py --input_format unite --output_format lstm --input_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/filtered_data.txt --output_file ${ATIS_data}act_int_${checkpoint}_lstm_${size}.txt --add_org --org_file ${ATIS_data}train_${size}_
#    python preprocess/get_evaluate_data.py --input_format unite --output_format bert --input_file ../result/atis/act_int_${size}/checkpoint-${checkpoint}/filtered_data.txt --output_file ${ATIS_data}act_int_${checkpoint}_bert_${size}.txt --add_org --org_file ${ATIS_data}train_${size}_
#done
#
##4. evaluate data
#for (( checkpoint = 1; checkpoint <= 5; checkpoint=checkpoint+1 ))
#do
#    for (( seed=0; seed <= 8080; seed=seed+2020 ))
#    do
#        cd ${current_pth}
#        python ../../Bi-LSTM_PosTagger/postagger.py -tt --train_path ${ATIS_data}act_int_${checkpoint}_lstm_${size}.txt --dev_path ${ATIS_data}val_lstm.txt --test_path ${ATIS_data}test_lstm.txt --label_set_path ../../Bi-LSTM_PosTagger/data/atis_vocab.txt --model ../../Bi-LSTM_PosTagger/model/ --output ../../Bi-LSTM_PosTagger/result/result.txt --script ../../Bi-LSTM_PosTagger/eval/conlleval.pl --gpu ${gpu_id} --max_epoch 300 --result_file ${result_folder}act_int_${checkpoint}_atis_${size}.txt --seed ${seed}
#        cp ${ATIS_data}act_int_${checkpoint}_bert_${size}.txt ../../slot_filling_and_intent_detection_of_SLU/data/atis-2/train
#        cd /sdc/htlin/Code/slot_filling_and_intent_detection_of_SLU/
#        python scripts/slot_tagging_and_intent_detection_with_pure_transformer.py --task_st NN --task_sc none --dataset atis --dataroot data/atis-2 --lr 5e-5 --dropout 0.1 --batchSize 32 --optim bertadam --max_norm 1 --experiment exp --deviceId ${gpu_id} --max_epoch 30 --st_weight 0.5 --pretrained_model_type bert --pretrained_model_name ../pretrained_model/bert_base_uncased/ --result_file ${result_folder}act_int_${checkpoint}_atis_${size}.txt --random_seed ${seed}
#    done
#done


# test for snips and size 327

# 1. finetune bart
size=13084
cd ${bart_folder}
#python run_act_to_raw.py --train_data_file ${SNIPS_data}train_${size}_raw.txt --train_intent_file ${SNIPS_data}train_${size}_intent.txt --train_label_file ${SNIPS_data}train_${size}_label.txt --output ../result/snips/act_${size}/  --cache_dir cache/ --do_train --num_train_epochs 3 --logging_steps 1000 --save_steps 0 --overwrite_output_dir --overwrite_cache --seed ${seed} --cuda_id ${gpu_id} --model_name_or_path ../../pretrained_model/bart_large/ --do_eval --eval_all_checkpoints --eval_data_file ${SNIPS_data}val_raw.txt --eval_intent_file ${SNIPS_data}val_intent.txt --eval_label_file ${SNIPS_data}val_label.txt
#
## 2. generate data
#for (( checkpoint = 1; checkpoint <= 5; checkpoint=checkpoint+1 ))
#do
#    python get_act_to_raw.py --gen_data_file ${SNIPS_data}train_${size}_raw.txt --gen_label_file ${SNIPS_data}train_${size}_label.txt --gen_intent_file ${SNIPS_data}train_${size}_intent.txt --slot_dict_file ${SNIPS_data}train_${size}_slot_dict.json --output_file ../result/snips/act_${size}/checkpoint-${checkpoint}/gen_data.txt --output_act_file ../result/snips/act_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --model_type bart --model_name_or_path ../result/snips/act_${size}/checkpoint-${checkpoint}/ --seed ${seed} --cuda_id ${gpu_id} --change_input
#    python filter_act_data.py --raw_file ${SNIPS_data}train_${size}_raw.txt --delex_file ${SNIPS_data}train_${size}_delex.txt --act_file ../result/snips/act_${size}/checkpoint-${checkpoint}/train_${size}_act.txt --gen_file ../result/snips/act_${size}/checkpoint-${checkpoint}/gen_data.txt --filter_file ../result/snips/act_${size}/checkpoint-${checkpoint}/filtered_data.txt --label_file ${SNIPS_data}train_${size}_label.txt
#done

#3. convert data
cd ${current_pth}
for (( checkpoint = 1; checkpoint <= 3; checkpoint=checkpoint+1 ))
do
    python preprocess/get_evaluate_data.py --input_format unite --output_format lstm --input_file ../result/snips/act_${size}/checkpoint-${checkpoint}/filtered_data.txt --output_file ${SNIPS_data}act_${checkpoint}_lstm_${size}.txt --add_org --org_file ${SNIPS_data}train_${size}_
    python preprocess/get_evaluate_data.py --input_format unite --output_format bert --input_file ../result/snips/act_${size}/checkpoint-${checkpoint}/filtered_data.txt --output_file ${SNIPS_data}act_${checkpoint}_bert_${size}.txt --add_org --org_file ${SNIPS_data}train_${size}_
done

#4. evaluate data
for (( checkpoint = 1; checkpoint <= 3; checkpoint=checkpoint+1 ))
do
    for (( seed=0; seed <= 8080; seed=seed+2020 ))
    do
        cd ${current_pth}
        python ../../Bi-LSTM_PosTagger/postagger.py -tt --train_path ${SNIPS_data}act_${checkpoint}_lstm_${size}.txt --dev_path ${SNIPS_data}val_lstm.txt --test_path ${SNIPS_data}test_lstm.txt --label_set_path ../../Bi-LSTM_PosTagger/data/snips_vocab.txt --model ../../Bi-LSTM_PosTagger/model/ --output ../../Bi-LSTM_PosTagger/result/result.txt --script ../../Bi-LSTM_PosTagger/eval/conlleval.pl --gpu ${gpu_id} --max_epoch 300 --result_file ${result_folder}act_int_${checkpoint}_snips_${size}.txt --seed ${seed}
        cp ${SNIPS_data}act_${checkpoint}_bert_${size}.txt ../../slot_filling_and_intent_detection_of_SLU/data/snips_lower/train
        cd /sdc/htlin/Code/slot_filling_and_intent_detection_of_SLU/
        python scripts/slot_tagging_and_intent_detection_with_pure_transformer.py --task_st NN --task_sc none --dataset snips --dataroot data/snips_lower --lr 5e-5 --dropout 0.1 --batchSize 32 --optim bertadam --max_norm 1 --experiment exp --deviceId ${gpu_id} --max_epoch 30 --st_weight 0.5 --pretrained_model_type bert --pretrained_model_name ../pretrained_model/bert_base_uncased/ --result_file ${result_folder}act_int_${checkpoint}_snips_${size}.txt --random_seed ${seed}
    done
done


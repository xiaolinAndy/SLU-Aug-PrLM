# SLU-Aug-PrLM
The code for Interspeech2021 paper "Augmenting Slot Values and Contexts for Spoken Language Understanding with Pretrained Models"

## Getting Started

### 1. Requirements

- Python 3.6
- Pytorch == 1.4.0
- transformers ==2.5.1
- tensorboard == 2.5.0

### 2. Resources

1. Download [bart-large](https://github.com/huggingface/transformers) pretrained model and put it into *resources/* folder.
2. Substitute the *modeling_bart.py* file in the transformer package with the given file  *utils/modeling_bart.py*

### 3. Training

#### Augmenting Slot Values

Refer to the bash file *utils/run_bart_single.sh* (You need to enter the directory according to your settings in the bash file)

1. First, choose the training dataset size by 

   ```
   size=327 # for snips you can choose {327, 1308, 13084}
   ```

2. Fine-tune the bart-large model with the given dataset.

   ```
   python run_delex_to_raw_single.py --train_data_file ${SNIPS_data}train_${size}_raw.txt --train_delex_file ${SNIPS_data}train_${size}_delex.txt --train_label_file ${SNIPS_data}train_${size}_label.txt --output ../result/snips/single_${size}/  --cache_dir cache/ --do_train --num_train_epochs 5 --logging_steps 1000 --save_steps 0 --overwrite_output_dir --overwrite_cache --seed ${seed} --cuda_id ${gpu_id} --model_name_or_path ../resources/bart_large
   ```

3. Run the fine-tuned model and generate augmented data.

   ```
   python get_delex_to_raw_single.py --gen_data_file ${SNIPS_data}train_${size}_raw.txt --gen_delex_file ${SNIPS_data}train_${size}_delex.txt --gen_label_file ${SNIPS_data}train_${size}_label.txt --output_file ../result/snips/single_${size}/checkpoint-${checkpoint}/gen_data.txt --output_delex_file ${SNIPS_data}train_${size}_single_delex.txt --output_label_file ${SNIPS_data}train_${size}_single_label.txt --model_type bart --model_name_or_path ../result/snips/single_${size}/checkpoint-${checkpoint}/ --seed ${seed} --cuda_id ${gpu_id}
   ```

4. Filter the unqualified data and save the augmented data in *../result/snips/single_\${size}/checkpoint-​\${checkpoint}/filtered_data.txt*

   ```
   python filter_single_data.py --delex_file ${SNIPS_data}train_${size}_single_delex.txt --raw_file ${SNIPS_data}train_${size}_raw.txt --label_file ${SNIPS_data}train_${size}_single_label.txt --gen_file ../result/snips/single_${size}/checkpoint-${checkpoint}/gen_data.txt --filter_file ../result/snips/single_${size}/checkpoint-${checkpoint}/filtered_data.txt --org_file ${SNIPS_data}train_${size}_
   ```

5. Use [LSTM](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) or [BERT](https://github.com/sz128/slot_filling_and_intent_detection_of_SLU) to test the result of augmented data.

#### Augmenting Contexts

Refer to the bash file *utils/run_bart_act.sh* (You need to enter the directory according to your settings in the bash file)

The process is consistent with the one in Augmenting Slot Values

### 4. Others

We also provide the link of other baselines methods for comparison

- [VAE](https://github.com/snipsco/automatic-data-generation)
- [Seq2seq](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU)

If you have any questions, please contact with haitao.lin@nlpr.ia.ac.cn

 
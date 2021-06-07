import torch
import os
import re
import argparse
import logging
import random
import pickle
import json
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.distributions.categorical import Categorical
from transformers import (
    WEIGHTS_NAME,
    PreTrainedTokenizer,
    PreTrainedModel,
    BartTokenizer,
    BartForMaskedLM,
    BartConfig
)

from BartExt import BartForGeneration, BartForSlotFiling, BartForSlotFilingCRF, BartForGenerationTest
ACT_DELETE_RATIO = 0.2
ACT_REPLACE_RATIO = 0.5
SEQ_NUM = 5

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "bart": (BartConfig, BartForGenerationTest, BartTokenizer)
}

def get_token_label(tokenizer, lines, labels):
    tok_labels = []
    for line, label in zip(lines, labels):
        line = line.split()
        label = label.split()
        tok_label = ['O']
        for word, l in zip(line, label):
            tok = tokenizer.tokenize(' ' + word)
            tok_label += [l] * len(tok)
        tok_label.append('O')
        tok_labels.append(tok_label)
    return tok_labels

def convert_id_to_label(labels, slot_dict):
    new_dict = {v:k for k,v in slot_dict.items()}
    token_type_ids = []
    for sample in labels:
        type_id = new_dict[sample.item()]
        token_type_ids.append(type_id)
    return token_type_ids

def convert_label_to_id(labels, slot_dict):
    token_type_ids = []
    for sample in labels:
        type_id = [slot_dict[s] for s in sample]
        token_type_ids.append(type_id)
    return token_type_ids

def split_intent(intent):
    intent = list(intent)
    for i, c in enumerate(intent):
        if c.isupper():
            intent[i] = ' ' + c.lower()
    intent = ''.join(intent).strip()
    return intent

def change_act(act, slot_dict):
    # act:[(s1, v1), ...]
    new_act = []
    for slot, value in act:
        if slot not in slot_dict.keys():
            print(slot)
            assert False
        prob = random.random()
        if prob >= ACT_REPLACE_RATIO:
            value = random.choice(slot_dict[slot])
        new_act.append([slot, value])
    return new_act


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path, label_path, intent_path, slot_dict_path, output_act_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            examples = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        with open(label_path, encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        with open(intent_path, encoding="utf-8") as f:
            intents = [line.strip() for line in f.readlines()]
        with open(slot_dict_path, encoding="utf-8") as f:
            slot_dict = json.load(f)
        new_acts, act_values = self.get_act_data(examples, intents)
        self.acts = tokenizer.batch_encode_plus(new_acts, max_length=block_size)["input_ids"]
        with open(output_act_path, 'w', encoding="utf-8") as f:
            for act in act_values:
                for i in range(SEQ_NUM):
                    f.write('\t'.join(act) + '\n')

    def get_act_data(self, examples, intents):
        new_acts = []
        act_values = []
        for example, intent in zip(examples, intents):
            example = example.split()
            intent_str = split_intent(intent)
            act = []
            for w in example:
                if w[0] == '_':
                    act.append(w)
            act_str = ' ' + intent_str + ' ( '
            act_value = []
            for i, slot in enumerate(act):
                slot_str = re.sub('_', ' ', slot[1:-1])
                slot_str = re.sub('\.', ' ', slot_str)
                act_str += slot + ' = ' + slot_str
                if i < len(act) - 1:
                    act_str += ' ; '
                act_value.extend([slot])
            act_str += ')'
            new_acts.append(act_str)
            act_values.append(act_value)
            if not act_str:
                assert False
        return new_acts, act_values

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, i):
        return torch.tensor(self.acts[i], dtype=torch.long)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# generated data form: line: text '\t' label
def gen_data(model, tokenizer, args):
    eval_dataset = LineByLineTextDataset(tokenizer, args, file_path=args.gen_data_file, label_path=args.gen_label_file,
                                         intent_path=args.gen_intent_file, slot_dict_path=args.slot_dict_file, output_act_path=args.output_act_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size

    # Note that DistributedSampler samples randomly
    def collate(examples: List[torch.Tensor]):
        input_ids = pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
        data = input_ids
        return data

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    gen_seqs = []
    gen_strs = []

    # Eval!
    logger.info("***** Running generation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch
        batch_size = inputs.shape[0]
        inputs = inputs.to(args.device)

        with torch.no_grad():
            outputs, outputs_att_id = model.generate(input_ids=inputs, max_length=40, temperature=0.8, top_p=0.9,
                           bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id,
                           eos_token_ids=tokenizer.eos_token_id, num_return_sequences=SEQ_NUM, repetition_penalty=1.4,
                                                    num_beams=1)
            # print(tokenizer.convert_ids_to_tokens(inputs[0]))
            # print(tokenizer.convert_ids_to_tokens(outputs[0]))
            # exit()
            for i in range(batch_size):
                # the original sentence
                #input_token = tokenizer.convert_ids_to_tokens(inputs[i])
                #input_str = tokenizer.decode(inputs[i], skip_special_tokens=True)
                for j in range(SEQ_NUM):
                    #output_token = tokenizer.convert_ids_to_tokens(outputs[seq_num*i+j])
                    output_str = tokenizer.decode(outputs[SEQ_NUM*i+j])
                    gen_strs.append(output_str)
    with open(args.output_file, "w") as writer:
        for data in gen_strs:
            writer.write(data + '\n')

# added function
def get_slot_dict(slot_file):
    slot_dict = {}
    count = 0
    with open(slot_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            slot_dict[line] = count
            count += 1
    return slot_dict

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--gen_data_file", default=None, type=str, required=True, help="data source for generating."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The generated result file",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    # Other parameters
    parser.add_argument(
        "--mlm_probability", type=float, default=0.2, help="Ratio of tokens to mask for prediction"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=2020, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--cuda_id", type=int, default=0, help="gpu id")
    parser.add_argument("--slot_type_list", type=str, default="../data/snips/slot_type_list.txt", help="")
    parser.add_argument("--gen_label_file", default='../data/snips/train_1308_label.txt', type=str)
    parser.add_argument("--gen_intent_file", default='../data/snips/train_1308_intent.txt', type=str)
    parser.add_argument("--output_act_file", default='../data/snips/train_1308_act.txt', type=str)
    parser.add_argument("--slot_dict_file", default='../data/snips/train_1308_slot_dict.json', type=str)
    parser.add_argument("--change_input", action="store_true", help='whether to change the input of dialogue acts')

    args = parser.parse_args()

    device = torch.device(args.cuda_id if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.slot_dict = get_slot_dict(args.slot_type_list)
    num_class = len(args.slot_dict.keys())
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    config.slot_size = num_class
    config.output_attentions = True
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    gen_data(model, tokenizer, args)

if __name__ == '__main__':
    main()
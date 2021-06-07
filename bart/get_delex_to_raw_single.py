import torch
import os
import re
import argparse
import logging
import random
import pickle
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


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path, label_path, delex_path, intent_path, output_delex_path, output_label_path, output_intent_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            examples = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        with open(label_path, encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        with open(delex_path, encoding="utf-8") as f:
            delexs = [line.strip() for line in f.readlines()]
        with open(intent_path, encoding="utf-8") as f:
            intents = [line.strip() for line in f.readlines()]
        new_delexs, new_examples, new_labels, new_intents = self.get_train_data(labels, delexs, examples, intents, args)
        with open(output_delex_path, 'w', encoding="utf-8") as f:
            for line in new_delexs:
                f.write(line + '\n')
        with open(output_label_path, 'w', encoding="utf-8") as f:
            for line in new_labels:
                f.write(line + '\n')
        with open(output_intent_path, 'w', encoding="utf-8") as f:
            for line in new_intents:
                f.write(line + '\n')
        self.examples = tokenizer.batch_encode_plus(new_examples, add_special_tokens=True, max_length=block_size)["input_ids"]
        self.delexs = tokenizer.batch_encode_plus(new_delexs, add_special_tokens=True, max_length=block_size)["input_ids"]
        print(len(new_delexs))

    def get_train_data(self, labels, delexs, examples, intents, args):
        new_delexs, new_examples, new_labels, new_intents = [], [], [], []
        for label, delex, example, intent in zip(labels, delexs, examples, intents):
            label = label.split()
            example = example.split()
            delex = delex.split()
            assert len(label) == len(example)
            for d in delex:
                if d[0] == '_':
                    d = d[1:-1]
                    new_delex = []
                    for i, l in enumerate(label):
                        if l[0] == 'B' and l[2:] == d:
                            if args.use_mask:
                                new_delex.append('<mask>')
                            else:
                                new_delex.append('_')
                                new_delex.append(re.sub('_', ' ', d))
                                new_delex.append('_')
                        elif l[0] == 'I' and l[2:] == d:
                            continue
                        else:
                            new_delex.append(example[i])
                    new_delexs.append(' '.join(new_delex))
                    new_examples.append(' '.join(example))
                    new_labels.append(' '.join(label))
                    new_intents.append(intent)
        return new_delexs, new_examples, new_labels, new_intents

    def __len__(self):
        return len(self.delexs)

    def __getitem__(self, i):
        return torch.tensor(self.delexs[i], dtype=torch.long)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# the map between delex and input_token
def get_delex_index(delex, input_token, tokenizer):
    # return index:{'_name_': [index_1, index_2, ...]}
    count = 0
    index = {}
    for w in delex:
        if w[0] == '_':
            index[w] = []
        w_d = re.sub('_', ' ', w)
        if w_d[0] != ' ':
            w_d = ' ' + w_d
        tokens = tokenizer.tokenize(w_d)
        #print(tokens)
        for t in tokens:
            if t != '_':
                if w[0] == '_':
                    index[w].append(count)
                count += 1
    token_count = 0
    for w in input_token:
        if w != '<pad>':
            token_count += 1
    #print(count, token_count)
    assert count == token_count
    return index

def get_output_att(output_token, attn_value):
    # attn_value: tgt_len * src_len
    output_text = []
    output_att = []
    tmp_str = ''
    tmp_att_value = 0
    for i, w in enumerate(output_token):
        if w == '<pad>':
            pass
        elif w == '<s>':
            output_text.append(w)
            output_att.append(attn_value[i])
        elif w == '</s>':
            output_text.append(tmp_str)
            output_att.append(tmp_att_value)
            output_text.append(w)
            output_att.append(attn_value[i])
        elif w[0] == 'Ġ':
            if tmp_str != '':
                output_text.append(tmp_str)
                output_att.append(tmp_att_value)
            tmp_str = w[1:]
            tmp_att_value = attn_value[i]
        else:
            tmp_str += w
    assert len(output_att) == len(output_text)

    return output_text, output_att

def get_slot_pred(att_value, delex_index_map):
    max_ind = torch.argmax(att_value).item()
    delex_rev_map = {}
    for k, vs in delex_index_map.items():
        for v in vs:
            delex_rev_map[v] = k
    if max_ind in delex_rev_map.keys():
        return delex_rev_map[max_ind]
    else:
        return 'O'

    '''max_value = 0
    tmp_slot = ''
    for k, vs in delex_index_map.items():
        tmp_value = 0.
        for v in vs:
            tmp_value += att_value[v].item()
        tmp_value /= len(vs)
        if max_value < tmp_value:
            max_value = tmp_value
            tmp_slot = k
    return tmp_slot'''

def convert_label(label, delex_index_map):
    last_label = 'O'
    slot_stat = {}
    for i in range(len(label)):
        if label[i][0] == '_' and label[i] == last_label:
            label[i] = 'I-' + last_label[1:-1]
        elif label[i][0] == '_':
            last_label = label[i]
            label[i] = 'B-' + last_label[1:-1]
            if last_label[1:-1] in slot_stat:
                #print(label)
                return 'error'
            else:
                slot_stat[last_label[1:-1]] = 0
        else:
            last_label = 'O'
    if len(delex_index_map.keys()) != len(slot_stat.keys()):
        return 'error'

    return label

def get_output_label(input_pth, gen_data, tokenizer):
    output_data = []
    with open(input_pth, 'r') as f:
        delex_data = [line.strip().split() for line in f.readlines()]
    for delex, otpt in zip(delex_data, gen_data):
        """ delex: [i want to watch _movie_number_ _movie_type_ movie]
            input_token: [<s> i want to watch movie number movie type movie </s> <pad> <pad>]
            output_token: [<s> i want to watch 1 horror movie </s> <pad> <pad>]
            attn_id: [1 2 3 .... 0 2 0]
            
            return: [[i want to watch 1 horror movie], 
                     [O O O O B-movie_number B-movie_type O]]
        """
        input_token, output_token, attn_id = otpt
        delex = ['<s>'] + delex + ['</s>']
        delex_index_map = get_delex_index(delex, input_token, tokenizer)
        output_txt, output_att = get_output_att(output_token, attn_id)
        #print(delex_index_map, output_txt, output_att)
        none_slot_index = 0
        label = []
        none_slot_word = []
        for s in delex:
            if s[0] != '_':
                none_slot_word.append(s)

        # att + matching
        gen_index = 0
        gen = output_txt
        flag = False
        for i, w in enumerate(delex):
            if w == gen[gen_index]:
                gen_index += 1
                label.append('O')
                continue
            elif delex[i][0] == '_' and delex[i+1][0] != '_':
                while gen_index < len(gen):
                    if gen[gen_index] == delex[i + 1]:
                        break
                    label.append(delex[i])
                    gen_index += 1
                # no matching slot, filter out
                else:
                    break
            # continuous slot
            elif delex[i][0] == '_' and delex[i + 1][0] == '_':
                while gen_index < len(gen):
                    slot = get_slot_pred(output_att[gen_index], delex_index_map)
                    if slot == delex[i]:
                        gen_index += 1
                        label.append(delex[i])
                    elif slot == delex[i+1]:
                        break
                    else:
                        flag = True
                        break
                else:
                    break
                if flag:
                    break
            else:
                break
        else:
            #print(count)
            label = convert_label(label, delex_index_map)
            if label != 'error':
                output_data.append([gen[1:-1], label[1:-1]])

        # pure att
        '''for i, w in enumerate(output_txt):
            if w == none_slot_word[none_slot_index]:
                none_slot_index += 1
                label.append('O')
            else:
                slot = get_slot_pred(output_att[i], delex_index_map)
                label.append(slot)
        if none_slot_index == len(none_slot_word):
            label = convert_label(label, delex_index_map)
            if label != 'error':
                output_data.append([output_txt[1:-1], label[1:-1]])'''

        #exit()
    print(len(output_data))
    return output_data


# generated data form: line: text '\t' label
def gen_data(model, tokenizer, args):
    eval_dataset = LineByLineTextDataset(tokenizer, args, file_path=args.gen_data_file, label_path=args.gen_label_file,
                                     delex_path=args.gen_delex_file, intent_path=args.gen_intent_file, output_delex_path=args.output_delex_file,
                                     output_label_path=args.output_label_file, output_intent_path=args.output_intent_file)
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
        seq_num = 1
        #print(inputs[0])
        #print(tokenizer.convert_ids_to_tokens(inputs[0]))

        with torch.no_grad():
            outputs, outputs_probs = model.generate(input_ids=inputs, max_length=40, temperature=1, top_p=0.9,
                           bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id,
                           eos_token_ids=tokenizer.eos_token_id, num_return_sequences=seq_num, repetition_penalty=1.4,
                                                    num_beams=1)
            # print(tokenizer.convert_ids_to_tokens(inputs[0]))
            # print(tokenizer.convert_ids_to_tokens(outputs[0]))
            # exit()
            # for prob, index in outputs_probs:
            #     print(prob)
            #     print(tokenizer.convert_ids_to_tokens(index))
            #exit()
            for i in range(batch_size):
                # the original sentence
                #input_token = tokenizer.convert_ids_to_tokens(inputs[i])
                #input_str = tokenizer.decode(inputs[i], skip_special_tokens=True)
                for j in range(seq_num):
                    #output_token = tokenizer.convert_ids_to_tokens(outputs[seq_num*i+j])
                    output_str = tokenizer.decode(outputs[seq_num*i+j], skip_special_tokens=True)
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
    parser.add_argument("--crf", action="store_true", help="use crf")
    parser.add_argument("--share_dec", action="store_true", help="share same decoder for lm and slot tagging")
    parser.add_argument("--filt_att", action="store_true", help="filter the generated data by attention")
    parser.add_argument("--gen_delex_file", default='../data/snips/train_1308_delex.txt', type=str)
    parser.add_argument("--gen_label_file", default='../data/snips/train_1308_label.txt', type=str)
    parser.add_argument("--gen_intent_file", default='../data/snips/train_1308_intent.txt', type=str)
    parser.add_argument("--output_delex_file", default='../data/snips/train_1308_single_delex.txt', type=str)
    parser.add_argument("--output_label_file", default='../data/snips/train_1308_single_label.txt', type=str)
    parser.add_argument("--output_intent_file", default='../data/snips/train_1308_single_intent.txt', type=str)
    parser.add_argument("--use_mask", action="store_true")

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
    if args.crf:
        model.crf_layer.set_device(args.device)
    model.to(args.device)
    gen_data(model, tokenizer, args)

if __name__ == '__main__':
    main()
import sys
import pickle
import random
import argparse
import re

SEED = 2020

random.seed(SEED)
slot_change = {}
with open('../data/atis/slot_vocab_change.txt') as f:
    for line in f.readlines():
        new_s, old_s = line.strip().split('\t')
        slot_change[new_s] = old_s

def get_count(slot, text):
    count = 0
    for w in text:
        if w == slot:
            count += 1
    return count

def convert_label(label):
    last_label = 'O'
    for i in range(len(label)):
        if label[i][0] == '_' and label[i] == last_label:
            label[i] = 'I-' + last_label[1:-1]
        elif label[i][0] == '_':
            last_label = label[i]
            label[i] = 'B-' + last_label[1:-1]
        else:
            last_label = 'O'
    return label

def match_value(slot, value, raw, label):
    flag = False
    new_label = label.copy()
    compare_count = 0
    # atis
    #slot = slot_change['_'.join(slot)]
    # snips
    slot = '_'.join(slot)
    for i, w in enumerate(raw):
        if w == value[compare_count] and label[i] == 'O':
            if compare_count == 0:
                new_label[i] = 'B-' + slot
            else:
                new_label[i] = 'I-' + slot
            compare_count += 1
            if compare_count == len(value):
                flag = True
                break
        elif w == value[0] and label[i] == 'O':
            new_label = label.copy()
            new_label[i] = 'B-' + slot
            compare_count = 1
        else:
            new_label = label.copy()
            compare_count = 0
    return new_label, flag

def filter_data(acts, raws):
    gen_data = []
    for act, raw in zip(acts, raws):
        raw = re.sub('<s>|</s>|<pad>', '', raw)
        raw = raw.split()
        act = act.split()
        for w in raw:
            if w[0] == '_' and w not in act:
                break
        else:
            for slot in act:
                if slot not in raw:
                    break
                count = get_count(slot, raw)
                if count > 1:
                    break
            else:
                gen_data.append(' '.join(raw))
    print('matching data: ', len(gen_data))
    return gen_data

def get_slot_value_dict(src, tgt):
    new_dict = {}
    for org, gen in zip(src, tgt):
        delex = []
        slot_type = ''
        str = ''
        for w in org:
            if w == '_' and str == '':
                str = '_'
            elif w == '_' and str != '':
                delex.append(str)
                slot_type = str
                str = ''
            elif str != '':
                str += w + '_'
            else:
                delex.append(w)
        org = delex
        org = ['<s>'] + org + ['</s>']
        gen = ['<s>'] + gen + ['</s>']
        gen_index = 0
        slot_value = ''
        for i, w in enumerate(org):
            if w == gen[gen_index]:
                gen_index += 1
                continue
            elif org[i][0] == '_' and org[i+1][0] != '_':
                while gen_index < len(gen):
                    if gen[gen_index] == org[i + 1]:
                        break
                    slot_value += gen[gen_index] + ' '
                    gen_index += 1
                # no matching slot, filter out
                else:
                    break
            # continuous slot
            else:
                break
        else:
            if slot_type and slot_value:
                if slot_type in new_dict.keys():
                    new_dict[slot_type].add(slot_value.strip())
                else:
                    new_dict[slot_type] = [slot_value.strip()]
                    new_dict[slot_type] = set(new_dict[slot_type])
    for k, v in new_dict.items():
        new_dict[k] = list(v)
    #print(new_dict)
    return new_dict

def delex_text(txt, label):
    txt = txt.split()
    label = label.split()
    delex = []
    assert len(txt) == len(label)
    for t, l in zip(txt, label):
        if l[0] == 'B':
            delex.append('_' + l[2:] + '_')
        elif l == 'O':
            delex.append(t)
        else:
            continue
    return ' '.join(delex)


def clean_repeat(raw_data, delex_data, gen_data):
    data_dict = {}
    final_data = []
    # for txt in raw_data:
    #     data_dict[txt] = ''
    # all same delex delete
    for i, txt in enumerate(delex_data):
        data_dict[txt] = ''
    for txt in gen_data:
        if txt not in data_dict.keys():
            data_dict[txt] = ''
            final_data.append(txt)
    print('different data: ', len(final_data))
    # extract 1308 samples
    random.shuffle(final_data)
    return final_data[:len(raw_data)]

def slot_replace(delex_data, label_data, slot_dict):
    gen_data = []
    for delex, label in zip(delex_data, label_data):
        #delex = delex.split()
        new_text, new_label = [], []
        for w in delex:
            if w[0] == '_':
                value = random.choice(slot_dict[w])
                new_text += value.split()
                new_label += ['B-' + w[1:-1]]
                new_label += ['I-' + w[1:-1]] * (len(value.split()) - 1)
            else:
                new_text.append(w)
                new_label.append('O')
            assert len(new_label) == len(new_text)
        gen_data.append([' '.join(new_text), ' '.join(new_label)])
    return gen_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_file', type=str, help='original raw file')
    parser.add_argument('--delex_file', type=str, help='original delex file')
    parser.add_argument('--act_file', type=str, help='dialogue act file')
    parser.add_argument('--gen_file', type=str, help='generated data file')
    parser.add_argument('--filter_file', type=str, help='filtered data file')
    args = parser.parse_args()

    with open(args.raw_file, 'r') as f:
        raw_data = [line.strip() for line in f.readlines()]
    with open(args.delex_file, 'r') as f:
        delex_data = [line.strip() for line in f.readlines()]
    with open(args.act_file, 'r') as f:
        act_data = [line.strip() for line in f.readlines()]
    with open(args.gen_file, 'r') as f:
        augment_data = [line.strip() for line in f.readlines()]
    gen_data = filter_data(act_data, augment_data)
    gen_data = clean_repeat(raw_data, delex_data, gen_data)

    with open(args.filter_file, 'w') as f:
        for txt in gen_data:
            f.write(txt + '\n')



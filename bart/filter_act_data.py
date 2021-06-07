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

def too_short(raw, tmp_values):
    tmp_values = ' '.join(tmp_values).split()
    if len(raw) <= len(tmp_values) + 1:
        return True
    else:
        return False

def filter_data(acts, raws, slot_dict):
    slot_values_dict = []
    for k, v in slot_dict.items():
        slot_values_dict += v
    gen_data = []
    for act, raw in zip(acts, raws):
        raw = raw.split()
        label = ['O'] * len(raw)
        act = act.split('\t')
        assert len(act) % 2 == 0
        slot_values = []
        tmp_values = []
        for i in range(int(len(act)/2)):
            slot_values.append([act[2*i].split(), act[2*i+1].split()])
            tmp_values.append(act[2*i+1])
        flag = False
        for v in slot_values_dict:
            if v not in tmp_values and v in raw:
                for v_all in tmp_values:
                    if v in v_all:
                        break
                else:
                    flag = True
                    break
        if too_short(raw, tmp_values):
            flag = True
        if flag:
            continue
        for slot, value in slot_values:
            label, flag = match_value(slot, value, raw, label)
            if not flag:
                break
        else:
            gen_data.append([' '.join(raw), ' '.join(label)])
    print('matching data: ', len(gen_data))
    return gen_data

def get_slot_dict(data):
    # data = [raw, label, intent]
    slot_dict = {}
    for raw, label in data:
        raw = raw.split()
        label = label.split()
        tmp_str = ''
        tmp_label = ''
        for i, word in enumerate(raw):
            if label[i][0] == 'B':
                if tmp_str:
                    slot_dict[tmp_label].add(tmp_str)
                tmp_str = word
                tmp_label = label[i][2:]
                if tmp_label not in slot_dict.keys():
                    slot_dict[tmp_label] = set()
            elif label[i][0] == 'I':
                tmp_str = tmp_str + ' ' + word
            else:
                if tmp_str:
                    slot_dict[tmp_label].add(tmp_str)
                tmp_str = ''
                tmp_label = ''
        if tmp_str:
            slot_dict[tmp_label].add(tmp_str)
    for k,v in slot_dict.items():
        slot_dict[k] = list(v)
    return slot_dict

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
    for i, txt in enumerate(raw_data):
        data_dict[txt] = ''
    for txt, label in gen_data:
        if txt not in data_dict.keys():
            data_dict[txt] = label
            final_data.append([txt, label])
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

def transfer_separate_data(raw_txt, label_txt):
    with open(raw_txt, 'r') as f:
        raw_data = [line.strip() for line in f.readlines()]
    with open(label_txt, 'r') as f:
        label_data = [line.strip() for line in f.readlines()]
    data = []
    for t, l in zip(raw_data, label_data):
        data.append([t, l])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_file', type=str, help='original raw file')
    parser.add_argument('--delex_file', type=str, help='original delex file')
    parser.add_argument('--act_file', type=str, help='dialogue act file')
    parser.add_argument('--gen_file', type=str, help='generated data file')
    parser.add_argument('--filter_file', type=str, help='filtered data file')
    parser.add_argument('--label_file', type=str, help='label file')
    args = parser.parse_args()

    with open(args.raw_file, 'r') as f:
        raw_data = [line.strip() for line in f.readlines()]
    with open(args.delex_file, 'r') as f:
        delex_data = [line.strip() for line in f.readlines()]
    with open(args.act_file, 'r') as f:
        act_data = [line.strip() for line in f.readlines()]
    with open(args.gen_file, 'r') as f:
        augment_data = [line.strip() for line in f.readlines()]

    data = transfer_separate_data(args.raw_file, args.label_file)
    slot_dict = get_slot_dict(data)

    gen_data = filter_data(act_data, augment_data, slot_dict)
    gen_data = clean_repeat(raw_data, delex_data, gen_data)


    with open(args.filter_file, 'w') as f:
        for txt, label in gen_data:
            f.write(txt + '\t' + label + '\n')



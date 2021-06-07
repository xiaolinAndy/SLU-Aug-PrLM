import sys
import pickle
import random
import argparse

SEED = 2020

random.seed(SEED)
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

def transfer_separate_data(raw_txt, label_txt, intent_txt, delex_txt):
    with open(raw_txt, 'r') as f:
        raw_data = [line.strip() for line in f.readlines()]
    with open(label_txt, 'r') as f:
        label_data = [line.strip() for line in f.readlines()]
    with open(intent_txt, 'r') as f:
        intent_data = [line.strip() for line in f.readlines()]
    with open(delex_txt, 'r') as f:
        delex_data = [line.strip() for line in f.readlines()]
    data = []
    for t, l, i, d in zip(raw_data, label_data, intent_data, delex_data):
        data.append([t, l, i, d])
    return data

def get_slot_dict(data):
    # data = [raw, label, intent]
    slot_dict = {}
    for raw, label, _, _ in data:
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

# src is delexicalized and tgt is raw text
def filter_data(src, tgt, lbl, intents):
    gen_data = []
    count = 1
    print(len(src))
    for org, gen, lb, it in zip(src, tgt, lbl, intents):
        delex = []
        new_label = []
        str = ''
        for w in org:
            if w == '_' and str == '':
                str = '_'
            elif w == '_' and str != '':
                delex.append(str)
                str = ''
            elif str != '':
                str += w + '_'
            else:
                delex.append(w)
        for l in lb:
            if l[0] == 'I' and '_' + l[2:] + '_' in delex:
                continue
            else:
                new_label.append(l)
        org = delex
        org = ['<s>'] + org + ['</s>']
        gen = ['<s>'] + gen + ['</s>']
        lb = ['O'] + new_label + ['O']
        assert len(org) == len(lb)
        gen_index = 0
        label = []
        value, target_label = [], ''
        for i, w in enumerate(org):
            if w == gen[gen_index]:
                gen_index += 1
                label.append(lb[i])
                continue
            elif org[i][0] == '_' and org[i+1][0] != '_':
                while gen_index < len(gen):
                    if gen[gen_index] == org[i + 1]:
                        break
                    label.append(org[i])
                    value.append(gen[gen_index])
                    target_label = org[i]
                    gen_index += 1
                # no matching slot, filter out
                else:
                    break
            # continuous slot
            else:
                break
        else:
            #print(count)
            if target_label:
                label = convert_label(label)
                value = ' '.join(value)
                gen_data.append([gen[1:-1], label[1:-1], it, value, target_label[1:-1]])
        count += 1
    print(len(gen_data))
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

def clean_repeat(raw_data, gen_data, slot_dict):
    data_dict = {}
    final_data = []
    wanted_data = []
    count = 0.
    for txt in raw_data:
        data_dict[txt] = ['', '']
    for txt, label, intent, value, target_label in gen_data:
        txt = ' '.join(txt)
        label = ' '.join(label)
        if value not in slot_dict[target_label]:
            count += 1
            wanted_data.append([txt, label, intent])
        if txt not in data_dict.keys():
            data_dict[txt] = [label, intent]
    for k, v in data_dict.items():
        if v[0] != '':
            final_data.append([k, v[0], v[1]])
    print(len(final_data))
    print(count, len(gen_data), count/len(gen_data))
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
    parser.add_argument('--label_file', type=str, help='dialogue act file')
    parser.add_argument('--gen_file', type=str, help='generated data file')
    parser.add_argument('--filter_file', type=str, help='filtered data file')
    parser.add_argument('--intent_file', type=str, default='', help='filtered data file')
    parser.add_argument('--org_file', type=str, help='data prefix')
    args = parser.parse_args()


    # delex_file = sys.argv[1]  # delex file
    # org_raw_file = sys.argv[2] # org raw file
    # org_label_file = sys.argv[3]  # org label file
    # augment_file = sys.argv[4]  # gen_file
    # filter_file = sys.argv[5]
    # org_delex_file = sys.argv[6]
    # intent_file = sys.argv[7]
    with open(args.delex_file, 'r') as f:
        delex_data = [line.strip().split() for line in f.readlines()]
    with open(args.raw_file, 'r') as f:
        raw_data = [line.strip() for line in f.readlines()]
    with open(args.label_file, 'r') as f:
        label_data = [line.strip().split() for line in f.readlines()]
    with open(args.gen_file, 'r') as f:
        augment_data = [line.strip().split() for line in f.readlines()]
    # with open(org_delex_file, 'r') as f:
    #     org_delex_data = [line.strip().split() for line in f.readlines()]
    if args.intent_file == '':
        intent_data = [''] * len(augment_data)
    else:
        with open(intent_file, 'r') as f:
            intent_data = [line.strip() for line in f.readlines()]

    raw_text = args.org_file + 'raw.txt'
    label_text = args.org_file + 'label.txt'
    intent_text = args.org_file + 'intent.txt'
    delex_text = args.org_file + 'delex.txt'
    org_data = transfer_separate_data(raw_text, label_text, intent_text, delex_text)
    slot_dict = get_slot_dict(org_data)
    gen_data = filter_data(delex_data, augment_data, label_data, intent_data)
    gen_data = clean_repeat(raw_data, gen_data, slot_dict)
    #slot_dict = get_slot_value_dict(delex_data, augment_data)
    #gen_data = slot_replace(org_delex_data, label_data, slot_dict)
    if args.intent_file == '':
        with open(args.filter_file, 'w') as f:
            for txt, label, _ in gen_data:
                f.write(txt + '\t' + label + '\n')
    else:
        with open(args.filter_file, 'w') as f:
            for txt, label, intent in gen_data:
                f.write(txt + '\t' + label + '\t' + intent + '\n')



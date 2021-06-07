'''
Prepare data for pretrained_slu experiment
At present, we support ATIS and snips data
Data size are 1/40 and 1/10 and full for the original dataset
The output files include: raw texts, delexicalized texts, slot labels, intents
'''
import random
import json

atis_data_folder = '../../data/atis/raw/'
snips_data_folder = '../../data/snips/raw/'
atis_output_folder = '../../data/atis/'
snips_output_folder = '../../data/snips/'
proportion = [1/40, 1/10, 1]
SEED = 2020
random.seed(SEED)

def get_slice(data, index):
    new_data = []
    for i in index:
        new_data.append(data[i])
    return new_data

def get_train_data(data_folder, output_folder, proportion):
    # train_data
    with open(data_folder + 'train/seq.in', 'r') as f:
        raw_text = [line.strip() for line in f.readlines()]
    with open(data_folder + 'train/seq.out', 'r') as f:
        slot_label = [line.strip() for line in f.readlines()]
    with open(data_folder + 'train/label', 'r') as f:
        intents = [line.strip() for line in f.readlines()]
    assert len(raw_text) == len(slot_label)
    assert len(raw_text) == len(intents)

    delex_text = []
    for text, label, intent in zip(raw_text, slot_label, intents):
        text = text.split()
        label = label.split()
        delex = []
        assert len(text) == len(label)
        for i, w in enumerate(label):
            if w == 'O':
                delex.append(text[i])
            elif w[0] == 'B':
                delex.append('_' + w[2:] + '_')
            else:
                continue
        delex_text.append(' '.join(delex))

    # split data
    dataset_size = len(raw_text)
    index_seq = list(range(dataset_size))
    random.shuffle(index_seq)
    for p in proportion:
        new_size = int(dataset_size * p)
        new_index = index_seq[:new_size]
        name_pre = output_folder + 'train_' + str(new_size) + '_'
        # raw_text
        with open(name_pre + 'raw.txt', 'w') as f:
            for data in get_slice(raw_text, new_index):
                f.write(data + '\n')
        # delex_text
        with open(name_pre + 'delex.txt', 'w') as f:
            for data in get_slice(delex_text, new_index):
                f.write(data + '\n')
        # label_text
        with open(name_pre + 'label.txt', 'w') as f:
            for data in get_slice(slot_label, new_index):
                f.write(data + '\n')
        # intent
        with open(name_pre + 'intent.txt', 'w') as f:
            for data in get_slice(intents, new_index):
                f.write(data + '\n')

def get_val_data(data_folder, output_folder, type):
    # val_data
    if type == 'val':
        prefix = 'valid'
    elif type == 'test':
        prefix = 'test'
    with open(data_folder + prefix + '/seq.in', 'r') as f:
        raw_text = [line.strip() for line in f.readlines()]
    with open(data_folder + prefix + '/seq.out', 'r') as f:
        slot_label = [line.strip() for line in f.readlines()]
    with open(data_folder + prefix + '/label', 'r') as f:
        intent = [line.strip() for line in f.readlines()]
    assert len(raw_text) == len(slot_label)
    assert len(raw_text) == len(intent)

    delex_text = []
    for text, label, int in zip(raw_text, slot_label, intent):
        text = text.split()
        label = label.split()
        delex = []
        assert len(text) == len(label)
        for i, w in enumerate(label):
            if w == 'O':
                delex.append(text[i])
            elif w[0] == 'B':
                delex.append('_' + w[2:] + '_')
            else:
                continue
        delex_text.append(' '.join(delex))

    name_pre = output_folder + type + '_'
    # raw_text
    with open(name_pre + 'raw.txt', 'w') as f:
        for data in raw_text:
            f.write(data + '\n')
    # delex_text
    with open(name_pre + 'delex.txt', 'w') as f:
        for data in delex_text:
            f.write(data + '\n')
    # label_text
    with open(name_pre + 'label.txt', 'w') as f:
        for data in slot_label:
            f.write(data + '\n')
    # intent
    with open(name_pre + 'intent.txt', 'w') as f:
        for data in intent:
            f.write(data + '\n')

def get_slot_intent(data_folder, output_folder):
    with open(data_folder + 'train/seq.out', 'r') as f:
        slot_label = [line.strip() for line in f.readlines()]
    with open(data_folder + 'train/label', 'r') as f:
        intents = [line.strip() for line in f.readlines()]
    with open(data_folder + 'valid' + '/seq.out', 'r') as f:
        slot_label += [line.strip() for line in f.readlines()]
    with open(data_folder + 'valid' + '/label', 'r') as f:
        intents += [line.strip() for line in f.readlines()]
    with open(data_folder + 'test' + '/seq.out', 'r') as f:
        slot_label += [line.strip() for line in f.readlines()]
    with open(data_folder + 'test' + '/label', 'r') as f:
        intents += [line.strip() for line in f.readlines()]

    # get slot labels:
    slots = set()
    for label in slot_label:
        label = label.split()
        for l in label:
            if l != 'O':
                slots.add(l[2:])
    with open(output_folder + 'slot_vocab.txt', 'w') as f:
        for s in slots:
            f.write(s + '\n')

    # get intent labels:
    ints = set(intents)
    with open(output_folder + 'intent_vocab.txt', 'w') as f:
        for s in ints:
            f.write(s + '\n')

# two sentences have the same slots
def get_parallel_data(delex_file, para_file):
    with open(delex_file, 'r') as f:
        delex_text = [line.strip().split() for line in f.readlines()]
    slot_delex_dict = {}
    for delex in delex_text:
        slots = []
        for w in delex:
            if w[0] == '_':
                slots.append(w)
        slots.sort()
        slots = '#'.join(slots)
        if slots not in slot_delex_dict:
            slot_delex_dict[slots] = [' '.join(delex)]
        else:
            slot_delex_dict[slots].append(' '.join(delex))

    para_data = []
    count = 0
    for slot, texts in slot_delex_dict.items():
        texts = list(set(texts))
        if len(texts) > 1:
            count += len(texts)
            for i in range(len(texts)):
                for j in range(len(texts)):
                    if i != j:
                        para_data.append([texts[i], texts[j]])
    with open(para_file, 'w') as f:
        for data in para_data:
            f.write(data[0] + '\t' + data[1] + '\n')
    print(count, len(delex_text))

def get_slot_dict(output_folder, sizes):
    for size in sizes:
        name_pre = output_folder + 'train_' + str(size) + '_'
        with open(name_pre + 'raw.txt', 'r') as f:
            raw_data = [line.strip() for line in f.readlines()]
        with open(name_pre + 'label.txt', 'r') as f:
            label_data = [line.strip() for line in f.readlines()]
        with open(name_pre + 'intent.txt', 'r') as f:
            intent_data = [line.strip() for line in f.readlines()]
        with open(name_pre + 'delex.txt', 'r') as f:
            delex_data = [line.strip() for line in f.readlines()]
        data = []
        for t, l, i, d in zip(raw_data, label_data, intent_data, delex_data):
            data.append([t, l, i, d])

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
        with open(name_pre + 'slot_dict.json', 'w') as f:
            json.dump(slot_dict, f, ensure_ascii=False)

if __name__ == '__main__':
    # get_train_data(atis_data_folder, atis_output_folder, proportion)
    # get_val_data(atis_data_folder, atis_output_folder, 'val')
    # get_val_data(atis_data_folder, atis_output_folder, 'test')
    # get_slot_intent(atis_data_folder, atis_output_folder)
    # print('Finish getting atis data !')
    # get_train_data(snips_data_folder, snips_output_folder, proportion)
    # get_val_data(snips_data_folder, snips_output_folder, 'val')
    # get_val_data(snips_data_folder, snips_output_folder, 'test')
    # get_slot_intent(snips_data_folder, snips_output_folder)
    # print('Finish getting snips data !')
    # para
    # get_parallel_data('../../data/snips/train_1308_delex.txt', '../../data/snips/train_1308_para.txt')
    get_slot_dict(atis_output_folder, [111, 447, 4478])
    get_slot_dict(snips_output_folder, [327, 1308, 13084])






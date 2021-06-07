'''
prepare data for evaluating
support source data in (raw, label) separate file or in a single mixed file
support evaluate data in conll format (for LSTM) or mixed format (for BERT)
'''

import sys
import re
import argparse

# separate data
def transfer_separate_data(raw_txt, label_txt, intent_txt):
    with open(raw_txt, 'r') as f:
        raw_data = [line.strip() for line in f.readlines()]
    with open(label_txt, 'r') as f:
        label_data = [line.strip() for line in f.readlines()]
    with open(intent_txt, 'r') as f:
        intent_data = [line.strip() for line in f.readlines()]
    data = []
    for t, l, i in zip(raw_data, label_data, intent_data):
        data.append([t, l, i])
    return data

# generated data
def transfer_gen_data(txt):
    data = []
    with open(txt, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            #assert len(line) == 2
            data.append(line)
    return data

# add original data and new data together
def gen_conll_data(org_data, new_data, output_file):
    with open(output_file, 'w') as f:
        for count, data in enumerate(org_data + new_data):
            utter = data[0].split()
            label = data[1].split()
            if len(utter) != len(label):
                print(utter, label)
                print(len(utter), len(label))
                print(count)
            assert len(utter) == len(label)
            for i in range(len(utter)):
                f.write(utter[i] + '\t' + label[i] + '\n')
            f.write('\n')

# clean txt to become lower case and delete quotation marks
def clean_txt(txt_list):
    new_txt = []
    for word in txt_list:
        if len(word) > 1:
            word = re.sub("'", "", word.lower())
        else:
            word = word.lower()
        new_txt.append(word)
    return new_txt

# uncleaned data to conll format
def snips2conll(utter='seq.in', label='seq.out', out='snips.conll'):
    data = {'utter':[], 'label':[]}
    with open(utter, 'r') as f:
        for line in f.readlines():
            data['utter'].append(line.strip().split())
    with open(label, 'r') as f:
        for line in f.readlines():
            data['label'].append(line.strip().split())
    with open(out, 'w') as f:
        for i in range(len(data['utter'])):
            utter = clean_txt(data['utter'][i])
            label = data['label'][i]
            if len(utter) != len(label):
                print(utter, label)
                exit()
            for j in range(len(utter)):
                if label[j] == 'B-timeRange':
                    label[j] = 'B-timerange'
                if label[j] == 'I-timeRange':
                    label[j] = 'I-timerange'
                f.write(utter[j] + '\t' + label[j] + '\n')
            f.write('\n')

def get_lstm_data(gen_data, output_file):
    with open(output_file, 'w') as f:
        for count, data in enumerate(gen_data):
            utter = data[0].split()
            label = data[1].split()
            if len(utter) != len(label):
                print(utter, label)
                print(len(utter), len(label))
                print(count)
            assert len(utter) == len(label)
            for i in range(len(utter)):
                f.write(utter[i] + '\t' + label[i] + '\n')
            f.write('\n')

def get_bert_data(gen_data, output_file):
    with open(output_file, 'w') as f:
        for count, data in enumerate(gen_data):
            utter = data[0].split()
            label = data[1].split()
            if len(utter) != len(label):
                print(utter, label)
                print(len(utter), len(label))
                print(count)
            assert len(utter) == len(label)
            for i in range(len(utter)):
                f.write(utter[i] + ':' + label[i] + ' ')
            if len(data) == 3:
                f.write('<=> ' + str(data[2]) + '\n')
            else:
                if 'snips' in output_file:
                    f.write('<=> AddToPlaylist\n')
                else:
                    f.write('<=> atis_flight\n')

def convert_conll_to_bert(conll_data, bert_data):
    data = []
    with open(conll_data, 'r') as f:
        text, label = [], []
        for line in f.readlines():
            if line == '\n':
                data.append([text, label])
                text, label = [], []
            else:
                line = line.strip().split('\t')
                text.append(line[0])
                label.append(line[1])
    with open(bert_data, 'w') as f:
        for count, data in enumerate(data):
            utter = data[0]
            label = data[1]
            if len(utter) != len(label):
                print(utter, label)
                print(len(utter), len(label))
                print(count)
            assert len(utter) == len(label)
            for i in range(len(utter)):
                f.write(utter[i] + ':' + label[i] + ' ')
            f.write('<=> AddToPlaylist\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_format', type=str, help='separate: for raw text and label text, unite: for mixed data')
    parser.add_argument('--output_format', type=str, help='lstm or bert')
    parser.add_argument('--input_file', type=str, help='spearate for prefix(train_1000_) and unite for file name')
    parser.add_argument('--output_file', type=str, help='output txt')
    parser.add_argument('--add_org', action="store_true", help='integrate generated data and original data together')
    parser.add_argument('--org_file', type=str, default=None, help='original data')
    args = parser.parse_args()

    if args.input_format == 'separate':
        raw_text = args.input_file + 'raw.txt'
        label_text = args.input_file + 'label.txt'
        intent_text = args.input_file + 'intent.txt'
        gen_data = transfer_separate_data(raw_text, label_text, intent_text)
    elif args.input_format == 'unite':
        gen_data = transfer_gen_data(args.input_file)
    else:
        print('Wrong input format !')

    if args.add_org:
        raw_text = args.org_file + 'raw.txt'
        label_text = args.org_file + 'label.txt'
        intent_text = args.org_file + 'intent.txt'
        org_data = transfer_separate_data(raw_text, label_text, intent_text)
        gen_data += org_data

    if args.output_format == 'lstm':
        get_lstm_data(gen_data, args.output_file)
    elif args.output_format == 'bert':
        get_bert_data(gen_data, args.output_file)


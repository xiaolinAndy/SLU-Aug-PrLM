import csv

def write_csv(data, filename):
    with open(filename, 'w') as _file:
        writer = csv.writer(_file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)

def get_csv_data(data_pth):
    raw_pth = data_pth + '_raw.txt'
    label_pth = data_pth + '_label.txt'
    delex_pth = data_pth + '_delex.txt'
    intent_pth = data_pth + '_intent.txt'
    data = [['utterance', 'labels', 'delexicalised', 'intent']]
    with open(raw_pth, 'r') as f:
        raw = [line.strip() for line in f.readlines()]
    with open(label_pth, 'r') as f:
        label = [line.strip() for line in f.readlines()]
    with open(delex_pth, 'r') as f:
        delex = [line.strip() for line in f.readlines()]
    with open(intent_pth, 'r') as f:
        intent = [line.strip() for line in f.readlines()]
    for r, l, d, i in zip(raw, label, delex, intent):
        data.append([r, l, d, i])
    write_csv(data, data_pth + '.csv')

if __name__ == '__main__':
    # train_pth = '../../data/snips/train_13084'
    # get_csv_data(train_pth)
    # val_pth = '../../data/snips/val'
    # get_csv_data(val_pth)
    train_pth = '../../data/atis/train_4478'
    get_csv_data(train_pth)
    val_pth = '../../data/atis/val'
    get_csv_data(val_pth)

import re

if __name__ == '__main__':
    with open('../../data/atis/slot_vocab.txt', 'r') as f:
        slots = [line.strip() for line in f.readlines()]
    with open('../../data/atis/slot_vocab_change.txt', 'w') as f:
        for slot in slots:
            new_slot = re.sub('\.', '_', slot)
            f.write(new_slot + '\t' + slot + '\n')
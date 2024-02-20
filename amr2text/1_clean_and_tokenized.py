import re

source_file_paths = ['data/src-train.txt', 'data/src-val.txt', 'data/src-test.txt']
target_file_paths = ['data/tgt-train.txt', 'data/tgt-val.txt', 'data/tgt-test.txt']


def process(sent, flag):
    if flag == 'source':
        sent = re.sub('\)', ' ) ', sent)
        sent = re.sub('\(', ' ( ', sent)
        sent = re.sub('\.', ' . ', sent)
        sent = re.sub('_', ' _ ', sent)
        sent = re.sub(',', ' , ', sent)
        sent = re.sub(' {2,}', ' ', sent)
    sent = sent.lower()
    return sent


for file_path in source_file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        sents = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    sents = [process(sent, 'source') for sent in sents]

    with open(file_path.replace('.txt', '_post.txt'), 'w', encoding='utf-8') as f:
        for line in sents:
            f.write(line.strip() + '\n')


for file_path in target_file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        sents = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    sents = [process(sent, 'target') for sent in sents]

    with open(file_path.replace('.txt', '_post.txt'), 'w', encoding='utf-8') as f:
        for line in sents:
            f.write(line.strip() + '\n')
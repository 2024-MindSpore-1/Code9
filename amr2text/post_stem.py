from bleu import BLEU
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
lemmatizer = WordNetLemmatizer()  

bleu = BLEU()

# gt_path = 'data/logic/tgt-val_post.txt'
input_representation = 'tree_treelstm'
pred_name = 'origin'

path = 'preds\\{}\\{}\\{{logic-model_step_10000.pt}}.pred.txt'.format(input_representation, pred_name)

print(path)


def stem(line):
    stem_line = []
    for word, tag in pos_tag(line.split()):
        if tag.startswith('NN'):
            stem_line.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            stem_line.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            stem_line.append(lemmatizer.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            stem_line.append(lemmatizer.lemmatize(word, pos='r'))
        else:
            stem_line.append(word)
    
    stem_line = [word for word in stem_line if word not in ['be', 'a', 'the', 'an']]
    return ' '.join(stem_line)


with open(path, 'r', encoding='utf-8') as f:
    data = [stem(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

print(len(data))


with open(path.replace('.txt', '_stem.txt'), 'w', encoding='utf-8') as f:
    for line in data:
        f.write(line + '\n')
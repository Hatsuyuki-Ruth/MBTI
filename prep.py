__author__ = 'Ruth Wang'


import re
import pickle as pkl


url_regex = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[^ ]*)*'
num_regex = '\d+'


labels = []
sents = []

with open('../mbti.csv', 'r') as f:
    _ = f.readline()
    line = f.readline()
    while line:
        label = line[:4]
        sent = line[7:-3]
        # sent = sent.replace('.|||', '. ')
        # sent = sent.replace('|||', '. ')
        sent = re.sub(url_regex, '', sent)
        sent = re.sub(num_regex, '', sent)
        sent = re.sub(' +', ' ', sent)
        sent = re.sub('^ *', '', sent)
        sent = re.sub('\|+', '|', sent)
        sent = sent.lower()
        labels.append(label)
        sent = sent.split('|')
        new_sent = []
        for i in sent:
            if re.search('[a-zA-Z]', i):
                new_sent.append(i)
        sents.append(new_sent)
        line = f.readline()

label_id = {label: i for i, label in enumerate(set(labels))}

for i in range(len(labels)):
    labels[i] = label_id[labels[i]]

with open('labels.pkl', 'wb') as f:
    pkl.dump(labels, f)

with open('sents.pkl', 'wb') as f:
    pkl.dump(sents, f)

# print(labels[:10])
# print(len(set(labels)))
# print(sents[0])

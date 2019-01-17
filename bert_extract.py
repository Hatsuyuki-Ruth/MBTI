__author__ = 'Ruth Wang'


import pickle as pkl

import numpy as np
from bert_serving.client import BertClient


bc = BertClient()

with open('sents.pkl', 'rb') as f:
     sents = pkl.load(f)

n = len(sents)

embs = np.zeros((n, 50, 1024))

for i, person in enumerate(sents):
    cur_emb = bc.encode(person)
    cur_n = len(person)
    embs[i, :cur_n, :] = cur_emb
    # emb_mean = np.mean(cur_emb, axis=0)
    # emb_max = np.max(cur_emb, axis=0)
    # emb = np.hstack((emb_mean, emb_max))
    # embs.append(emb)

# embs = np.vstack(embs)

with open('embs.pkl', 'wb') as f:
    pkl.dump(embs, f)

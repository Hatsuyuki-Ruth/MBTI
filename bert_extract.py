__author__ = 'Ruth Wang'


import pickle as pkl

import numpy as np
from bert_serving.client import BertClient


bc = BertClient()

with open('sents.pkl', 'rb') as f:
     sents = pkl.load(f)

embs = []

for person in sents:
    cur_emb = bc.encode(person)
    emb_mean = np.mean(cur_emb, axis=0)
    emb_max = np.max(cur_emb, axis=0)
    emb = np.hstack((emb_mean, emb_max))
    embs.append(emb)

embs = np.vstack(embs)

with open('embs.pkl', 'wb') as f:
    pkl.dump(embs, f)

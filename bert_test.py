__author__ = 'Ruth Wang'


import pickle as pkl

from bert_serving.client import BertClient


bc = BertClient()

with open('sents.pkl', 'rb') as f:
    sents = pkl.load(f)

embs = bc.encode(sents)

with open('emb.pkl', 'wb') as f:
    pkl.dump(embs, f)

# with open('labels.pkl', 'rb') as f:
#     labels = pkl.load(f)

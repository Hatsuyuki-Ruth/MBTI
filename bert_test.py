__author__ = 'Ruth Wang'


import pickle as pkl

import numpy as np
import tensorflow as tf
from tensorflow import keras
from bert_serving.client import BertClient


bc = BertClient()

with open('sents.pkl', 'rb') as f:
    sents = pkl.load(f)

embs = bc.encode(sents)

with open('labels.pkl', 'rb') as f:
    labels = pkl.load(f)

print(embs.shape)
print(embs)

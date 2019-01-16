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

model = keras.Sequential()
model.add(keras.layers.Dense(1024, input_shape=(1024,), activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = embs[:1000]
x_train = embs[1000:]
y_val = labels[:1000]
y_train = labels[1000:]

history = model.fit(x_train,
                    y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

import pdb
pdb.set_trace()

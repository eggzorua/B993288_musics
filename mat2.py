import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
batch_size = 256

ic = models.load_model('saved_model\my_model')
npz = np.load('cqt.npz')
x = npz['spec']
y = npz['instr']
x = x.reshape((20488, 168, 87,1))

#y.reshape(1, -1)
ridx = list(range(len(x)))
random.shuffle(ridx)
test_size = len(ridx) // 10
test_ridx, train_ridx = ridx[:test_size], ridx[test_size:]

x2 = x[train_ridx]
y2 = y[train_ridx]
x3 = x[test_ridx]
y3 = y[test_ridx]

confusion = np.zeros([np.max(y) + 1] * 2, dtype=np.float32)


predictions = ic.predict(x3)
#for b in range(len(test_ridx)):
#    for a in range(174):
#        confusion[b,a]
answers = y3[:,1]
p2 = np.zeros(len(test_ridx),dtype=np.int)
for b in range(len(test_ridx)):
    p2[b] = predictions[b,:].argmax()
for b in range(len(test_ridx)):
    confusion[p2[b], answers[b]] += 1
#for p, a in zip(predictions, answers):
#    confusion[p2, a] += 1
    print("confusion[",p2[b],", ",answers[b],"]")

wrongs = []
gmlist = np.array([l.strip() for l in open('gm.list.txt')])

for i in range(np.max(y) + 1):
    if confusion[i, :].sum() - confusion[i, i] > 0 \
            or confusion[:, i].sum() - confusion[i, i] > 0: wrongs.append(i)

reduced = confusion[wrongs, :][:, wrongs]
reduced = reduced / np.maximum(reduced.sum(axis=0), 1)
reduced_list = gmlist[wrongs]

for i in range(len(reduced)):
    s = (-reduced[:, i]).argsort()
    conf_list = [reduced_list[j] for v, j in zip(reduced[s, i], s) if v > 0 and j != i]

    print(reduced_list[i], conf_list)

fig, ax = plt.subplots()
im = ax.imshow(reduced, interpolation='nearest')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(reduced.shape[0]), yticks=np.arange(reduced.shape[1]),
       xticklabels=reduced_list, yticklabels=reduced_list,
       xlabel='True label', ylabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.show()
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

npz = np.load('cqt.npz')
x = npz['spec']
y = npz['instr']
#y = y[:, 1] # y가 예상과 다르게 tuple 형식이어서 한번더 가공해줄겁니다.

ic = models.Sequential()
#ic.add(layers.Input(shape=([x.shape[1], x.shape[2]], np.max(y) + 1)))
ic.add(layers.Input(shape=(168, 87, 1)))# 168, 87, 173+1
print(x.shape[1])
print(x.shape[2])
print(np.max(y)+1)
#tf.expand_dims(x, axis=1).shape.as_list() #reshape나 이 구문으로 x를 가공해줘야 했는데, reshape가 더 다루기 편해 비활성화

#ic.add(layers.Input(shape=None)) #오류가 나는 구문입니다.
ic.add(layers.Conv2D(32, (3, 3),  activation='relu', name='layerone'))
ic.add(layers.Conv2D(32, (3, 3), activation='relu', name='layertwo'))
ic.add(layers.Conv2D(32, (24, 3), activation='relu', name='last'))
ic.add(layers.Flatten())
ic.add(layers.Dense(np.max(y) + 1, activation='softmax', name='outp'))
#ic.summary() # compile 전이면서 input shape값이 없으면 오류가 납니다.

ic.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('complied')
ic.summary()


z = [x, y]
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
#trainpertest = 900 #또다른 랜덤으로 나누는 방식이지만 위의 방식을 썼습니다.
#num1 = np.random.random((1000, trainpertest))
#num2 = np.random.random((1000, trainpertest))
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
# 체크포인트를 이용해 원하는 지점마다 불러올 수 있는 기능입니다. 필요하지 않아 주석 처리했습니다.

hist = ic.fit(x2, y2[:, 1], epochs=30, batch_size = 256) # epochs 1당 10분 가까이가 소요됩니다. 많은 학습은 어려웠습니다
print("training end!")
#for e in range(100): # 참고하였던 소스코드의 matplot쓰는 방식입니다. 역시 작동하지 않아 주석처리합니다.
#    for b in range(0, len(train_ridx), batch_size):
#       tr = train_ridx[b:b + batch_size]
#       loss_v = ic.fit(x[tr], y[tr, 0].astype(np.int32))
#        print('Epoch:{:04} Loss:{:.4}'.format(e, loss_v))
#
#    av, lv = [], []
#    for b in range(0, len(test_ridx), batch_size):
#        tr = test_ridx[b:b + batch_size]
#        acc_v, loss_v = ic.fit(x[tr], y[tr, 0].astype(np.int32))
#        av.append(acc_v)
#        lv.append(loss_v)
#    cur_acc = sum(av) / len(av)
#    print('Epoch:{:04} Test Acc:{:.4} Test Loss:{:.4}'.format(e, cur_acc, sum(lv) / len(lv)))
#    saver = tf.train.Saver()
#    saver.save('models/last')
#    if cur_acc > best_acc:
#        best_acc = cur_acc
#        saver.save('models/best')
batch_size = 64
test_loss, test_acc = ic.evaluate(x3, y3[:, 1], verbose=2)
print("test accuarcy is ", test_acc)


ic.save('saved_model/my_model') # 모델이 저장됩니다.


acc = hist.history['accuracy']
loss = hist.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy') # 학습, 테스트 정확도입니다.
plt.plot(epochs, test_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training loss') # 학습, 테스트 loss입니다.
plt.plot(epochs, test_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



# confusion = np.zeros([np.max(y) + 1] * 2, dtype=np.float32)
#
# for b in range(0, len(test_ridx), batch_size):
#     tr = test_ridx[b:b + batch_size]
#     predictions = ic.predict(x3[b,0:167,0:86,0:1])
#     answers = y3[:,b]
#     for p, a in zip(predictions, answers):
#         confusion[p, a] += 1
#
# wrongs = []
# gmlist = np.array([l.strip() for l in open('gm.list.txt')])
#
# for i in range(np.max(y) + 1):
#     if confusion[i, :].sum() - confusion[i, i] > 0 \
#             or confusion[:, i].sum() - confusion[i, i] > 0: wrongs.append(i)
#
# reduced = confusion[wrongs, :][:, wrongs]
# reduced = reduced / np.maximum(reduced.sum(axis=0), 1)
# reduced_list = gmlist[wrongs]
#
# for i in range(len(reduced)):
#     s = (-reduced[:, i]).argsort()
#     conf_list = [reduced_list[j] for v, j in zip(reduced[s, i], s) if v > 0 and j != i]
#
#     print(reduced_list[i], conf_list)
#
# fig, ax = plt.subplots()
# im = ax.imshow(reduced, interpolation='nearest')
# ax.figure.colorbar(im, ax=ax)
# ax.set(xticks=np.arange(reduced.shape[0]), yticks=np.arange(reduced.shape[1]),
#        xticklabels=reduced_list, yticklabels=reduced_list,
#        xlabel='True label', ylabel='Predicted label')
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
# plt.show()
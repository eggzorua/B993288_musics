# tensorflow 1으로 만들어진 코드입니다. 다운그레이해도 제대로 작동하지 않아 사용하지 않고,
# 코드 참고용으로만 남겨놓습니다.
import numpy as np
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disalbe_v2_behavior()

random.seed(777)

npz = np.load('cqt.npz')
x = npz['spec']
y = npz['instr']

class InstrumentClassfier:
    def __init__(self, input_shape, num_class, num_filters=[64, 128, 256]):
        print("wow! first dif start!")
        #self.input_x = tf.Variable(tf.ones(shape=[None, input_shape[0], input_shape[1]], name='input_x'))  # [batch, freq, time]
        #self.input_y = tf.Variable(tf.ones(shape=[None], name='input_y'))
        self.input_x = tf.compat.v1.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1]])  # [batch, freq, time]
        self.input_y = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.dropout_rate = tf.compat.v1.placeholder(tf.float32)
        print("wow! first dif end!")
        #self.dropout_rate = tf.Variable(tf.ones(tf.float32))

        self.filters = []

        x = tf.expand_dims(self.input_x, axis=-1)
        b_size = 1
        for i, size in enumerate(num_filters[:-1]):
            with tf.variable_scope('filter_{}'.format(i), reuse=tf.AUTO_REUSE):
                ft_w = tf.get_variable('w', shape=[3, 3, b_size, size],
                                       initializer=tf.contrib.layers.variance_scaling_initializer())
                ft_b = tf.get_variable('b', shape=[size])
                self.filters.append((ft_w, ft_b))
                x = tf.nn.relu(tf.nn.conv2d(x, ft_w, padding='SAME') + ft_b)
                x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                b_size = size

        with tf.variable_scope('filter_final', reuse=tf.AUTO_REUSE):
            ft_w = tf.get_variable('w', shape=[24, 3, num_filters[-2], num_filters[-1]],
                                   initializer=tf.contrib.layers.variance_scaling_initializer())
            ft_b = tf.get_variable('b', shape=[num_filters[-1]])
            self.filters.append((ft_w, ft_b))
            x = tf.nn.relu(tf.nn.conv2d(x, ft_w, padding='SAME') + ft_b)
            x = tf.reduce_max(x, axis=[1, 2])
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        with tf.variable_scope('fully', reuse=tf.AUTO_REUSE):
            fc_w = tf.get_variable('w', shape=[num_filters[-1], num_class],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.get_variable('b', shape=[num_class])
            logit = tf.matmul(x, fc_w) + fc_b
        self.prediction = tf.argmax(logit, axis=1, output_type=tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.input_y), tf.float32))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self.input_y))
        self.trainer = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, sess, x, y, test=False):
        return sess.run([self.acc if test else self.trainer, self.loss], feed_dict={
            self.input_x: x,
            self.input_y: y,
            self.dropout_rate: 0 if test else 0.5
        })

    def predict(self, sess, x):
        return sess.run(self.prediction, feed_dict={
            self.input_x: x,
            self.dropout_rate: 0
        })

ic = InstrumentClassfier([x.shape[1], x.shape[2]], np.max(y) + 1)

ridx = list(range(len(x)))
random.shuffle(ridx)

test_size = len(ridx) // 10
test_ridx, train_ridx = ridx[:test_size], ridx[test_size:]
batch_size = 256

best_acc = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(100):
        for b in range(0, len(train_ridx), batch_size):
            tr = train_ridx[b:b + batch_size]
            _, loss_v = ic.train(sess, x[tr], y[tr, 0].astype(np.int32))
            if (b // batch_size) % 10 == 0: print('Epoch:{:04} Loss:{:.4}'.format(e, loss_v))

        av, lv = [], []
        for b in range(0, len(test_ridx), batch_size):
            tr = test_ridx[b:b + batch_size]
            acc_v, loss_v = ic.train(sess, x[tr], y[tr, 0].astype(np.int32), test=True)
            av.append(acc_v)
            lv.append(loss_v)
        cur_acc = sum(av) / len(av)
        print('Epoch:{:04} Test Acc:{:.4} Test Loss:{:.4}'.format(e, cur_acc, sum(lv) / len(lv)))
        saver = tf.train.Saver()
        saver.save(sess, 'models/last')
        if cur_acc > best_acc:
            best_acc = cur_acc
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
a = tf.constant([1,2,3,4],tf.float32)
b = tf.constant([
                [1,2,3],
                [4,5,6]
                ],tf.float32)
c = tf.constant([
                 [[1,2],[3,4],[5,6]],
                 [[7,8],[9,10],[11,12]]
                ],tf.float32)

suma = tf.reduce_sum(a)
sumb1 = tf.reduce_sum(b,axis=0)
sumb2 = tf.reduce_sum(b,axis=1)
sumb3 = tf.reduce_sum(b,axis=(0,1))
sumc = tf.reduce_sum(c,axis=(0,1,2))
maxall = tf.argmax(b,axis=0)

with tf.Session() as sess:
    print(sess.run(suma))
    print(sess.run(sumb1))
    print(sess.run(sumb2))
    print(sess.run(sumb3))
    print(sess.run(sumc))
    print(sess.run(maxall))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

a = tf.constant([10,20],tf.float32)
b = tf.constant(
                [
                [1,2],
                [3,4]
                ],tf.float32
               )
c = tf.constant(
               [
               [
               [[1,2],[3,4]],
               [[5,6],[7,8]]
               ],
               [
               [[3,5],[7,9]],
               [[2,4],[4,6]]
               ]
               ],tf.float32
               )
add2d = tf.add(a,b)
add4d = tf.add(a,c)
with tf.Session() as sess:
    print(sess.run(add2d))
    print("-----------------\n")
    print(sess.run(add4d))


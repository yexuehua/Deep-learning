import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

one = tf.constant([1,2],tf.float32)
two = tf.constant([3,4],tf.float32)
a = tf.constant([
                [11,12,13],
                [14,15,16]
                ],tf.float32)
b = tf.constant([
                [4,5,6],
                [7,8,9]
                ],tf.float32)
c = tf.stack([a,b],axis=0)
d = tf.concat([one,two],axis=0)
sess = tf.Session()
print(sess.run([c,d]))
sess.close()

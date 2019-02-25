import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
a = tf.constant([
                [1,2,3],
                [5,6,7],
                [8,9,1]
                ],tf.float32
                )
s = tf.slice(a,[0,1],[2,2])
sess = tf.Session()
print(sess.run(s))

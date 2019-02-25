import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
a = tf.constant([
                [1,2,3],
                [4,5,6]
                ],tf.float32)
t = tf.transpose(a,perm=[1,0])
sess = tf.Session()
print(sess.run(t))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
a = tf.constant([
                [[1,2],[3,4],[5,6]],
                [[7,8],[9,10],[11,12]]
                ],tf.float32)
sess = tf.Session()
b = tf.reshape(a,[3,1,-1])
print(sess.run(b))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[2,None],name='x')
a = tf.constant([
                [1,2],
                [3,4],
                [5,6]
                ],tf.float32)
m = tf.matmul(a,x)
sess = tf.Session()
print(sess.run(m,feed_dict={x:np.array([[1,2],[3,4]],np.float32)}),"\n")
print(sess.run(m,feed_dict={x:np.array([[1],[2]],np.float32)}))
sess.close()

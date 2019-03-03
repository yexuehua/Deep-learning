import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np
import tensorflow as tf


x = tf.placeholder(tf.float32,(2,1))
w = tf.constant([[3,4]],tf.float32)
y = tf.matmul(w,x)
F = tf.pow(y,2)
grad = tf.gradients(F,x)
sess = tf.Session()
print(sess.run(grad,feed_dict={x:np.array([[2],[3]])}))
print("......")

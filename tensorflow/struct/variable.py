import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
a = tf.Variable(tf.constant([1,2],tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("the initail value of a\n")
print(sess.run(a))
sess.run(a.assign([10,20]))
print("the value of changed a")
print(sess.run(a))


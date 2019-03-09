import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
data_x
x = tf.placeholder(tf.float32,name="input")
J = tf.placeholder(tf.float32,name="lable")
w = tf.get_variable("weight",[1],initializer=tf.truncated_normal_initializer())
b = tf.get_variable("bias",[1],initializer=tf.constant_initializer())
J_predicted = w*x+b
loss = tf.squar(J-J_predicted,name="loss")
lr = 0.000001
optimizer = tf.train.GradientDescentOpitimizer(lr).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variable_initializer())
    for i in range(10):
        sess.run(optimizer,feed_dict={x:data_x,y:data_y})
        print("echo:{0}:{1}".format(i,sess.run(loss)))

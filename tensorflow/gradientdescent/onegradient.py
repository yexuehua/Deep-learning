import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

'''
x = tf.Variable(4.0,tf.float32)
y = tf.pow(x-1,2)
opti = tf.train.GradientDescentOptimizer(0.25).minimize(y)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3):
    sess.run(opti)
    print(sess.run(x))
'''


x = tf.Variable(15.0,tf.float32)
y = tf.pow(x-1,2)
value = np.arange(-15,17,0.01)
y_value = np.power(value-1,2.0)
plt.plot(value,y_value)

opti = tf.train.GradientDescentOptimizer(0.05).minimize(y)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
    sess.run(opti)
    if(i%10==0):
        v = sess.run(x)
        plt.plot(v,math.pow(v-1,2.0),'go')
        print("this is the %d times iterates,the x is:%f"%(i+1,v))
sess.close()
plt.show()










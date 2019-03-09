import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

sigma = 1
mu = 10
result = tf.random_normal([10,4,20,5],mu,sigma,tf.float32)
sess = tf.Session()
array = sess.run(result)
array1 = array.reshape([-1])
histogram,bins,patch = plt.hist(array1,25,facecolor='gray',alpha=0.5,normed=True)
x = np.arange(5,15,0.01)
y = 1.0/(math.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x-mu,2.0))/(2*math.pow(sigma,2))
plt.plot(x,y)
plt.show()

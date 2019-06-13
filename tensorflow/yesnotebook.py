                  Chaper1
#1.Ignoring warning by using flowing sentance:
 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

The log level of tensorflow including:info,warn,error,fatal
info(0)<warn(1)<error(2)<fatal(3)

#2.When we add two tensor,althought thier forms are different,tensorflow will aotomaticaly extend the tensor,which lack of element,to the same form with the other one:

value1 = tf.cosntant([
                    [1,2],
		    [3,4]
                   ],tf.float32)
value2 = tf.constant([10,10])
result = tf.add(value1,value2)
#3.transaction bettween ndarray and tensor,Numpy can store and processing mult-dimentional array,the core data struct is ndarry.tensorflow will convert tensor to ndarray by creating session,so that we can print the value of tensor:

t = tf.constant([1,2,3],tf.float32)
sess = tf.Session()
array = sess.run(t)#tensor convert to ndarray
t = tf.convert_to_tensor(array,tf.float32,name='t')#"type come back"

                 chapter2


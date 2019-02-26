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

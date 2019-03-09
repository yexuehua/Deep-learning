import tensorflow as tf
import matplotlib.pyplot as plt
image = tf.read_file("test.jpg",'r')
image_tensor = tf.image.decode_jpeg(image)
shape = tf.shape(image_tensor)
with tf.Session() as sess:
    print("the shape of image")
    print(sess.run(shape))
    array = sess.run(image_tensor)
    #array = image_tensor.eval(session=sess)
plt.imshow(array)
plt.show()

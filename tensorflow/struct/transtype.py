import tensorflow as tf
import matplotlib.pyplot as plt
image = tf.read_file("test.jpg",'r')
image_tensor = tf.image.decode_jpeg(image)
shape = tf.shape(image_tensor)
with tf.Session() as session:
    print("the shape of image")
    print(shape)
    array = image_tensor.eval()
plt.imshow(array)
plt.show()

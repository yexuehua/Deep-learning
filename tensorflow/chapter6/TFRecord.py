import tensorflow as tf
import numpy as np

# create a file
record = tf.python_io.TFRecordWriter("data.tfrecord")
# 3d ndarry for high=2,width=3,dim=4
array1 = np.array(
            [
            [[1,2,1,2],[3,4,2,9],[5,6,0,3]],
            [[7,8,1,6],[9,6,1,7],[1,2,5,9]]
            ],np.float32
            )
# 3d ndarray for (3,3,3)
array2 = np.array(
            [
            [[11,12,11],[13,14,12],[15.16,13]],
            [[17,18,11],[19,10,11],[11,12,15]],
            [[13,14,15],[18,11,12],[19,14,10]]
            ],np.float32
            )
# 3d ndarray for (2,2,3)
array3 = np.array(
            [
            [[21,23,21],[23,24,22]],
            [[27,28,24],[29,20,21]]
            ],np.float32
            )
# restore the 3 ndarray into a list
arrays = [array1,array2,array3]

# repeat to cope with every ndarray restored in the list
for array in arrays:
    # calculate the shape of ndarray
    height,width,depth = array.shape
    # convert the value of ndarray to byte type
    array_raw = array.tostring()

    feature = {
        'array_raw':
            tf.train.Feature(
                bytes_list = tf.train.BytesList(value = [array_raw])),
        'height':
            tf.train.Feature(
                int64_list = tf.train.Int64List(value = [height])),
        'width':
            tf.train.Feature(
                int64_list = tf.train.Int64List(value = [width])),
        'depth':
            tf.train.Feature(
                int64_list = tf.train.Int64List(value = [depth]))
    }
    features = tf.train.Features(feature = feature)
    example = tf.train.Example(features = features)
    record.write(example.SerializeToString())
record.close()
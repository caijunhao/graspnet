import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from nets import vgg

path = '/home/caijunhao/ros_ws/src/grasping/scripts/logs/checkpoints/vgg_16.ckpt'
images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='images')
num_classes = 1000
with slim.arg_scope(vgg.vgg_arg_scope()):
    net, end_points = vgg.vgg_16(inputs=images,
                                 num_classes=num_classes,
                                 is_training=True,
                                 dropout_keep_prob=1.0,
                                 scope='vgg_16')

variable_list = slim.get_variables_to_restore()
variable_map = {var.op.name: var for var in variable_list}
saver = tf.train.Saver(variable_map)

sess = tf.Session()
saver.restore(sess, path)

imgs = np.ones((1, 224, 224, 3), np.float32)
out = sess.run(net, feed_dict={images: imgs})
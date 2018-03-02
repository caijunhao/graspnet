from geometry_msgs.msg import Pose, Point, Quaternion

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from math import pi

from model import model, model_arg_scope, se_model
from baxter_api import BaxterAPI
from image_tools import ImageTools
import utils

import argparse
import rospy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


parser = argparse.ArgumentParser(description='grasping')
parser.add_argument('--topic_name', default='/cameras/left_hand_camera/image', type=str)
parser.add_argument('--limb', default='left', type=str)
parser.add_argument('--hover_distance', default=0.1563, type=float, help='meters')
parser.add_argument('--reach_distance', default=0.010, type=float, help='meters')
parser.add_argument('--img_id', default=0, type=int, help='image patch id to save')
parser.add_argument('--patch_size', default=224, type=int, help='image patch size to crop')
parser.add_argument('--num_patches', default=400, type=int)
parser.add_argument('--center_coor', default='/home/caijunhao/kinect/build/center.txt', type=str,
                    help='path to the file containing center coordinate of the objects')
parser.add_argument('--checkpoint_dir', default='/home/caijunhao/Desktop/domain_adaptation_for_grasping/logs_source_only',
                    type=str, help='Path where the checkpoint file locate.')
parser.add_argument('--output_path', default='test_result', type=str, help='output path.')  # source_only_result
args = parser.parse_args()
num_classes = 18

initial_orientation = Quaternion(x=0.0286820208316, y=0.999469262198, z=-0.00957069663929, w=0.0121217725021)
initial_coordinate = Point(x=0.5385790242396508, y=0.7734277417900787, z=0.23876878260620296)
initial_pose = Pose(position=initial_coordinate, orientation=initial_orientation)
origin = (0.35324136513088605, 0.6541424331347214, 0.01384132037567437)  # (0.39397914313440474, -0.6462901096088043)


def main():
    rospy.init_node(name='grasping', anonymous=True)
    f = open(args.output_path+'/grasping_label.txt', 'a+')
    img_id = len(f.readlines())
    pnp = BaxterAPI(args.limb, args.hover_distance, args.reach_distance)  # pick and place
    pnp.move_to_start(initial_pose)
    image_tools = ImageTools(img_id=img_id)

    images_t = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    images_t = images_t - [123.68, 116.779, 103.939]
    with slim.arg_scope(model_arg_scope()):
        net, end_points = model(inputs=images_t,
                                num_classes=num_classes,
                                is_training=False,
                                dropout_keep_prob=1.0,
                                scope='source_only')
    # with slim.arg_scope(model_arg_scope()):
    #     net, end_points = model(inputs=images_t,
    #                             num_classes=num_classes,
    #                             is_training=False,
    #                             dropout_keep_prob=1.0,
    #                             reuse=None,
    #                             scope='domain_adapt',
    #                             adapt_scope='target_adapt_layer',
    #                             adapt_dims=128)
        angle_index_t = tf.argmin(net, axis=1)

        saver = tf.train.Saver()
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        sess = tf.Session(config=session_config)
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
        print 'Successfully loading model.'

    t = 0
    success = 0

    while not rospy.is_shutdown() and t != 10:
        flag = 0
        while flag != 1:
            c_x, c_y = utils.get_center_coordinate_from_kinect(args.center_coor, args.limb)
            d_x = float(c_x) / 1000.0 + 0.07
            d_y = float(c_y) / 1000.0 + 0.03
            x = origin[0]+d_x
            y = origin[1]-d_y
            z = origin[2]
            flag = pnp.approach(Pose(position=Point(x, y, z), orientation=initial_orientation))
        rospy.sleep(0.7)
        utils.listener(args.topic_name, image_tools)
        coors, images = image_tools.sampling_image(args.patch_size, args.num_patches)
        scores = sess.run(net, feed_dict={images_t: images})
        patch, coor, new_coors = image_tools.resampling_image(scores, coors, args.patch_size)
        angle_index, scores= sess.run([angle_index_t, net], feed_dict={images_t: patch})
        angle_index = angle_index[0]
        print angle_index
        print scores
        grasp_angle = (angle_index * 10 - 90) * 1.0 / 180 * pi
        x -= float(coor[0] - 140) * 0.6 / 1000
        y -= float(coor[1] - 320) * 0.5 / 1000
        z = z
        pose = Pose(position=Point(x, y, z), orientation=initial_orientation)
        label = pnp.pick(pose,
                         image_tools,
                         args.output_path,
                         new_coors,
                         coor,
                         args.patch_size,
                         grasp_angle)
        success += 1 - label
        f.write('{:06d} {} {}\n'.format(image_tools.img_id - 1, angle_index, label))
        pnp.move_to_start(initial_pose)
        t += 1
    print success * 1.0 / t
    f.close()


if __name__ == '__main__':
    main()

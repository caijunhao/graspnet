from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion

import tensorflow as tf
import numpy as np
from math import pi

import argparse
import rospy

from nets import vgg
from baxter_api import BaxterAPI
from image_tools import ImageTools
import utils

parser = argparse.ArgumentParser(description='grasping')
parser.add_argument('--topic_name', default='/cameras/right_hand_camera/image', type=str)
parser.add_argument('--limb', default='right', type=str)
parser.add_argument('--hover_distance', default=0.1563, type=float, help='meters')
parser.add_argument('--reach_distance', default=0.001, type=float, help='meters')
parser.add_argument('--img_id', default=0, type=int, help='image patch id to save')
parser.add_argument('--patch_size', default=224, type=int, help='image patch size to crop')
parser.add_argument('--num_patches', default=400, type=int)
args = parser.parse_args()

initial_orientation = Quaternion(x=0.0286820208316, y=0.999469262198, z=-0.00957069663929, w=0.0121217725021)
initial_coordinate = Point(0.658153933314, -0.698397243796, 0.113631262747)
initial_pose = Pose(position=initial_coordinate, orientation=initial_orientation)
origin = (0.243271495648, 0.637174368871, 0.001)


def main():
    rospy.init_node(name='grasping', anonymous=True)
    pnp = BaxterAPI(args.limb, args.hover_distance, args.reach_distance)  # pick and place
    pnp.move_to_start(initial_pose)
    image_tools = ImageTools(img_id=args.img_id)

    while not rospy.is_shutdown():
        utils.listener(args.topic_name, image_tools.callback)
        # image_tools.display_image()


if __name__ == '__main__':
    main()

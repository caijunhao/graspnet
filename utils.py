from sensor_msgs.msg import Image

import rospy
import random


def listener(topic_name, image_tools):
    sub = rospy.Subscriber(topic_name, Image, image_tools.callback)
    while not image_tools.img_flag:
        rospy.sleep(0.1)  # 0.27
    image_tools.img_flag = False
    sub.unregister()


def get_center_coordinate_from_kinect(file_path, hand='right'):
    size = (1295, 685)
    coors = []
    while not len(coors):
        with open(file_path, 'r') as f:
            string = f.readline()[:-1]
        if string:
            coors = [int(coor) for coor in string.split(' ')]
            idx = random.randint(0, len(coors) / 2 - 1)
    if hand == 'right':
        return coors[idx * 2 + 1], size[0] - coors[idx * 2]
    else:
        return coors[idx * 2 + 1], coors[idx * 2]


def convert_to_baxter_coordinate(x, y, origin):
    delta_x = x * 1.0 / 1000  # height
    delta_y = y * 1.0 / 1000  # width
    # the order of baxter coordinate is width, height and depth
    b_x, b_y, b_z = origin[0] + delta_x, origin[1] - delta_y, origin[2]
    return b_x, b_y, b_z


def refine_baxter_coordinate(x, y, origin):
    delta_x = (x - 320) * 0.6 / 1000  # height
    delta_y = (y - 140) * 0.6 / 1000  # width
    # delta_z = (z * 10 - 90) * 1.0 / 180 * pi
    b_x, b_y, b_z = origin[0] - delta_x, origin[1] - delta_y, origin[2]
    return b_x, b_y, b_z

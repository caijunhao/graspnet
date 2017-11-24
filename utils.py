from sensor_msgs.msg import Image

import rospy


def listener(topic_name, callback):
    sub = rospy.Subscriber(topic_name, Image, callback)
    rospy.sleep(0.27)
    sub.unregister()


def get_center_coordinate_from_kinect():
    # todo: get center coordinate from saved txt file
    return 1, 2


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

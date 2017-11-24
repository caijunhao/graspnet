import struct
import copy

import numpy as np
from math import pi

import rospy
import baxter_interface

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
)

from std_msgs.msg import (
    Header,
)


class BaxterAPI(object):
    def __init__(self, limb, hover_distance=0.15, reach_distance=0.001, verbose=True):
        self._limb_name = limb
        self._hover_distance = hover_distance
        self._reach_distance = reach_distance
        self._verbose = verbose
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)

        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verified robot is enabled
        print 'Get robot state ...'
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print 'Enabling robot ...'
        self._rs.enable()

    def move_to_start(self, start_pose=None):
        print 'Moving the {0} arm to start pose ...'.format(self._limb_name)
        if not start_pose:
            start_angles = dict(zip(self._limb.joint_angles().keys(), [0]*7))
        else:
            start_angles = self.ik_request(start_pose)
        self.guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print 'Runing. Ctrl-c to quit'

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        # limb_joints = {}
        if resp_seeds[0] != resp.RESULT_INVALID:
            seed_str = {
                ikreq.SEED_USER: 'User Provided Seed',
                ikreq.SEED_CURRENT: 'Current Joint Angles',
                ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
            }.get(resp_seeds[0], 'None')
            if self._verbose:
                print "IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(seed_str)
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr('No Joint Angles provided for move_to_joint_positions. Staying put.')

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(0.1)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(0.1)

    def approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self.guarded_move_to_joint_position(joint_angles)

    def reach_out(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = self._reach_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self.guarded_move_to_joint_position(joint_angles)

    def retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self.guarded_move_to_joint_position(joint_angles)

    def servo_to_pose(self, pose):
        # servo down to release

        joint_angles = self.ik_request(pose)
        self.guarded_move_to_joint_position(joint_angles)

    def get_joint_angles(self):
        return self._limb.joint_angles()

    def pick(self, pose, end_point_angle):
        success = 0
        self.approach(pose)
        joint_angles = self.get_joint_angles()
        joint_angles['right_w2'] += end_point_angle
        self.guarded_move_to_joint_position(joint_angles)
        self.reach_out()
        self.gripper_close()
        self.retract()
        if self._gripper.position() > 6:
            success = 1
            joint_angles = self.get_joint_angles()
            joint_angles['right_w2'] = np.random.uniform(-pi, pi)
            self.guarded_move_to_joint_position(joint_angles)
            self.reach_out()
            self.gripper_open()
            self.retract()
        return success

    def pick_and_place(self, pose, end_point_angle, initial_pose):
        success = 0
        self.approach(pose)
        joint_angles = self.get_joint_angles()
        joint_angles['right_w2'] += end_point_angle
        self.guarded_move_to_joint_position(joint_angles)
        self.reach_out()
        self.gripper_close()
        self.retract()
        if self._gripper.position() > 6:
            success = 1
            self.servo_to_pose(initial_pose)
        self.gripper_open()
        return success

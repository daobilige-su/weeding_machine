#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rospy
import cashmerebot.msg
import actionlib
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int8MultiArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
import datetime
import time
import ros_numpy
from plot_marker import *

# params

class path_plan_action(object):
    _feedback = cashmerebot.msg.path_planFeedback()
    _result = cashmerebot.msg.path_planResult()

    def __init__(self):
        self._action_name = 'path_plan'
        self._as = actionlib.SimpleActionServer(self._action_name, cashmerebot.msg.path_planAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        # params:
        self.pc_roi = np.array([[0.2, 0.8], [-1, 1], [0, 1]])
        self.marker_pub = rospy.Publisher("path_plan_markers", MarkerArray, queue_size=5)

        self.surf_dist = 0.1
        self.lift_dist = 0.05
        self.angle_incr = 5  # deg
        self.ee_base_y_shift = 0.11

    def execute_cb(self, goal): # 0: stand still, 1: forward; 2, backward; 3, left; 4, right;

        success = False
        params = goal.params

        rospy.loginfo('%s: Executing, obtained goal location: (%f)' % (
            self._action_name, goal.params[0]))

        # read point cloud
        pc2_msg = rospy.wait_for_message('/front_depth_cam/pc2_ur5_base', PointCloud2, timeout=1.0)
        cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_msg).T  # 3xN

        # ROI
        cloud_roi = cloud.copy()
        # X limit
        cloud_roi = cloud_roi[:, cloud_roi[0, :] > self.pc_roi[0, 0]]
        cloud_roi = cloud_roi[:, cloud_roi[0, :] < self.pc_roi[0, 1]]
        # Y limit
        cloud_roi = cloud_roi[:, cloud_roi[1, :] > self.pc_roi[1, 0]]
        cloud_roi = cloud_roi[:, cloud_roi[1, :] < self.pc_roi[1, 1]]
        # Z limit
        cloud_roi = cloud_roi[:, cloud_roi[2, :] > self.pc_roi[2, 0]]
        cloud_roi = cloud_roi[:, cloud_roi[2, :] < self.pc_roi[2, 1]]

        # segment
        cloud_seg = cloud_roi.copy()

        # extract mid line in 2 points
        mid_line = np.array([[0.7, 0.7], [-0.9, 0.9], [0.4, 0.4]])


        # path plan
        cloud_ring = cloud_seg.copy()
        cloud_ring = cloud_ring[:, cloud_ring[1, :] > self.ee_base_y_shift-0.05]
        cloud_ring = cloud_ring[:, cloud_ring[1, :] < self.ee_base_y_shift+0.05]

        plot_pts(cloud_ring.T, self.marker_pub, 0, 'ur5_base')
        plot_traj(mid_line.T, self.marker_pub, 10, 'ur5_base', 0.02)

        ring_center = mid_line[:, 0].copy()
        ring_center[1] = self.ee_base_y_shift
        ring_center = ring_center.reshape((-1,))
        path_pose = self.plan_single_ring(cloud_ring, ring_center)

        # plot_traj(can_pose_ypr_valid[0:3, :].T, self.marker_pub, 2, 'ur5_base')
        plot_arrows(path_pose.T, self.marker_pub, 100, 'ur5_base')
        plot_traj(path_pose[0:3, :].T, self.marker_pub, 20, 'ur5_base')

        # store results
        path_list = path_pose.T.reshape((-1,)).tolist()

        self._result.path = path_list

        # return result
        success = True

        if success == True:
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)

    def plan_single_ring(self, cloud, center):
        cloud_2d = cloud[[0, 2], :].copy()
        center_2d = center[[0, 2]].copy()

        angle = np.arctan2(cloud_2d[1, :]-center_2d[1], -(cloud_2d[0, :]-center_2d[0]))
        angle_deg = np.rad2deg(angle)

        can_deg = range(90, -90-self.angle_incr, -self.angle_incr)
        can_num = len(can_deg)
        can_pts_num = np.zeros((1, can_num))
        can_pose_ypr = np.zeros((6, can_num))
        can_lift_pose_ypr = np.zeros((6, can_num))

        for idx in range(can_num):
            deg = can_deg[idx]

            deg_cloud_idx = (abs(angle_deg-deg)<(self.angle_incr/2.0))
            deg_cloud_pts = cloud_2d[:, deg_cloud_idx].copy()

            deg_cloud_pts_num = deg_cloud_pts.shape[1]
            can_pts_num[0, idx] = deg_cloud_pts_num

            if deg_cloud_pts_num<5:
                continue

            deg_cloud_pts_mean = np.mean(deg_cloud_pts, 1)
            dist_mean = np.sqrt(np.sum(np.power(center_2d - deg_cloud_pts_mean, 2)))
            dist_arm = dist_mean+self.surf_dist
            dist_arm_lift = dist_mean+self.surf_dist+self.lift_dist

            deg_cloud_pose = np.array([center_2d[0]-dist_arm*np.cos(np.deg2rad(deg)), center[1], center_2d[1]+dist_arm*np.sin(np.deg2rad(deg)), \
                                       0, np.deg2rad(deg), 0]).reshape((6,1))
            deg_cloud_lift_pose = np.array([center_2d[0] - dist_arm_lift * np.cos(np.deg2rad(deg)), center[1], center_2d[1] + dist_arm_lift * np.sin(np.deg2rad(deg)), \
                                            0, np.deg2rad(deg), 0]).reshape((6, 1))
            can_pose_ypr[:, idx:idx + 1] = deg_cloud_pose.copy()
            can_lift_pose_ypr[:, idx:idx + 1] = deg_cloud_lift_pose.copy()

        # only take the continuous section in the middle part
        can_num_half = int((can_num+1)/2)

        min_idx = can_num_half
        for idx in range(can_num_half):
            idx_in_vec = can_num_half-idx
            if can_pts_num[0, idx_in_vec]>5:
                min_idx = idx_in_vec
            else:
                break
        max_idx = can_num_half
        for idx in range(can_num_half):
            idx_in_vec = can_num_half+idx
            if can_pts_num[0, idx_in_vec]>5:
                max_idx = idx_in_vec
            else:
                break

        can_pose_ypr_valid = can_pose_ypr[:, min_idx:max_idx + 1].copy()
        can_lift_pose_ypr_valid = can_lift_pose_ypr[:, min_idx:max_idx + 1].copy()

        path_pose = np.block([[can_lift_pose_ypr_valid], [can_pose_ypr_valid]]).reshape((6,-1), order='F')

        return path_pose











if __name__ == '__main__':

    rospy.init_node('path_plan_server_node')
    server = path_plan_action()
    rospy.spin()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rospy
import cashmerebot.msg
import actionlib

from plot_marker import *
from transform_tools import *

from geometry_msgs.msg import Pose
import sys
import moveit_commander
from moveit_commander import MoveGroupCommander

from copy import deepcopy

# params


class manipulation_action(object):
    _feedback = cashmerebot.msg.manipulationFeedback()
    _result = cashmerebot.msg.manipulationResult()

    def __init__(self):
        self._action_name = 'manipulation'
        self._as = actionlib.SimpleActionServer(self._action_name, cashmerebot.msg.manipulationAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        # moveit related
        moveit_commander.roscpp_initialize(sys.argv)

        # 初始化需要使用move group控制的机械臂中的arm group
        self.arm = MoveGroupCommander('manipulator')

        # 当运动规划失败后，允许重新规划
        self.arm.allow_replanning(True)

        # 设置目标位置所使用的参考坐标系
        self.arm.set_pose_reference_frame('base_link')

        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.1)

        # 获取终端link的名称
        self.end_effector_link = self.arm.get_end_effector_link()

        # special poses
        self.home_pose = np.deg2rad(np.array([0, -150, 60, 90, 90, 0])).tolist()
        self.pre_exec_pose = np.deg2rad(np.array([0, -90, 60, 300, 270, 0])).tolist()

        # turn head z
        self.turn_head_z = 0.45

        # go to home
        self.arm.set_joint_value_target(self.home_pose)
        # 控制机械臂完成运动
        self.arm.go()
        rospy.sleep(1)


    def execute_cb(self, goal): # goal.path: list of poses in terms of trans+ypr

        success = False
        path_np = np.array(goal.path).reshape((-1, 6)).T
        path_pose_num = path_np.shape[1]

        rospy.loginfo('%s: obtained path, executing ...' % (self._action_name))

        # add poses to moveit planner

        # 初始化路点列表
        # waypoints = []

        # 获取当前位姿数据最为机械臂运动的起始位姿
        start_pose = self.arm.get_current_pose(self.end_effector_link).pose
        # 将初始位姿加入路点列表
        # waypoints.append(start_pose)

        # go to pre_exe pose
        # self.arm.set_joint_value_target(self.pre_exec_pose)
        # 控制机械臂完成运动
        # self.arm.go()
        # rospy.sleep(1)
        self.run_joint_target_one_by_one(self.pre_exec_pose)

        head_turned = False
        for n in range(path_pose_num):
            pose = path_np[:, n]

            m = transform_trans_ypr_to_matrix(pose)
            tool_m = m @ transform_trans_ypr_to_matrix(np.array([0, 0, 0, np.pi/2, 0, np.pi/2]))
            tool_pose_quat = transform_matrix_to_trans_quat(tool_m).reshape((-1,))

            # if the end-effector go below a threshold self.turn_head_z and it's the preparing pose,
            # then turn head for better posing
            if (tool_pose_quat[2]<self.turn_head_z) and (n%2==0) and (not head_turned):
                self.turn_head(tool_pose_quat)
                head_turned = True

            tar_pose = Pose()
            tar_pose.position.x = tool_pose_quat[0]
            tar_pose.position.y = tool_pose_quat[1]
            tar_pose.position.z = tool_pose_quat[2]
            tar_pose.orientation.x = tool_pose_quat[3]
            tar_pose.orientation.y = tool_pose_quat[4]
            tar_pose.orientation.z = tool_pose_quat[5]
            tar_pose.orientation.w = tool_pose_quat[6]

            self.arm.set_pose_target(tar_pose)
            # self.arm.plan()
            # self.arm.execute()
            self.arm.go()
            rospy.sleep(1)

        self.run_joint_target_one_by_one(self.home_pose)


        # return result
        success = True

        if success == True:
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)

    def run_joint_target_one_by_one(self, joint_target):
        joint_target_step = self.arm.get_current_joint_values()

        for n in range(6):
            joint_target_step[n] = joint_target[n]
            self.arm.set_joint_value_target(joint_target_step)
            # 控制机械臂完成运动
            self.arm.go()
            rospy.sleep(0.1)

    def turn_head(self, tool_pose_quat):
        # go to target pose
        tar_pose = Pose()
        tar_pose.position.x = tool_pose_quat[0]
        tar_pose.position.y = tool_pose_quat[1]
        tar_pose.position.z = tool_pose_quat[2]
        tar_pose.orientation.x = tool_pose_quat[3]
        tar_pose.orientation.y = tool_pose_quat[4]
        tar_pose.orientation.z = tool_pose_quat[5]
        tar_pose.orientation.w = tool_pose_quat[6]

        self.arm.set_pose_target(tar_pose)
        # self.arm.plan()
        # self.arm.execute()
        self.arm.go()
        rospy.sleep(1)

        # turn head
        joint_target_step = self.arm.get_current_joint_values()

        joint_target_step[1] = joint_target_step[1] - np.deg2rad(10)
        # joint_target_step[2] = joint_target_step[2] + np.deg2rad(10)

        if joint_target_step[3] < 0:
            joint_target_step[3] = joint_target_step[3] + np.deg2rad(180)
        else:
            joint_target_step[3] = joint_target_step[3] - np.deg2rad(180)
        if joint_target_step[4] < 0:
            joint_target_step[4] = joint_target_step[4] + np.deg2rad(180)
        else:
            joint_target_step[4] = joint_target_step[4] - np.deg2rad(180)
        if joint_target_step[5] < 0:
            joint_target_step[5] = joint_target_step[5] + np.deg2rad(180)
        else:
            joint_target_step[5] = joint_target_step[5] - np.deg2rad(180)

        self.run_joint_target_one_by_one(joint_target_step)

        # go to target pose again
        tar_pose = Pose()
        tar_pose.position.x = tool_pose_quat[0]
        tar_pose.position.y = tool_pose_quat[1]
        tar_pose.position.z = tool_pose_quat[2]
        tar_pose.orientation.x = tool_pose_quat[3]
        tar_pose.orientation.y = tool_pose_quat[4]
        tar_pose.orientation.z = tool_pose_quat[5]
        tar_pose.orientation.w = tool_pose_quat[6]

        self.arm.set_pose_target(tar_pose)
        # self.arm.plan()
        # self.arm.execute()
        self.arm.go()
        rospy.sleep(1)



if __name__ == '__main__':
    try:
        rospy.init_node('manipulation_server_node')
        server = manipulation_action()
        rospy.spin()
    except rospy.ROSInterruptException:
        # 关闭并退出moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

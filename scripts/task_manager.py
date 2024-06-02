#! /usr/bin/env python

import numpy as np
import cashmerebot.msg
import rospy
# from __future__ import print_function

# Brings in the SimpleActionClient
import actionlib

from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion, Twist
from visualization_msgs.msg import Marker
import rospy
import sys
from std_msgs.msg import String, Float32MultiArray
from cashmerebot.srv import TaskList, ConvCmd
from tf import transformations


class TaskManager:
    def __init__(self):
        # self.sub = rospy.Subscriber("chatter", String, self.callback)
        self.task_list = None  # np.zeros((20, 10))
        self.no_more_task_warned = 0
        self.task_sleep_rate = rospy.Rate(10)

        # TaskList service to update self.task_list
        self.task_list_srv = rospy.Service('TaskList', TaskList, self.update_task_list)
        rospy.loginfo('TaskList service ready')

        # sending conveyor position cmd
        rospy.wait_for_service('ConvCmd')
        rospy.loginfo('ConvCmd service connected.')
        self.conv_cmd_request = rospy.ServiceProxy('ConvCmd', ConvCmd)

        # path_plan client
        self.path_plan_client = actionlib.SimpleActionClient('path_plan', cashmerebot.msg.path_planAction)
        self.path_plan_client.wait_for_server()
        rospy.loginfo('path_plan_server connected.')

        # manipulation client
        self.manipulation_client = actionlib.SimpleActionClient('manipulation', cashmerebot.msg.manipulationAction)
        self.manipulation_client.wait_for_server()
        rospy.loginfo('manipulation_server connected.')

        # ready to start
        rospy.logwarn('Going to Start after 3s ...')
        rospy.sleep(3)
        rospy.logwarn('Started!')

    # when client want to update task_list, cancel the current jobs and update the
    def update_task_list(self, req):
        task_list = np.array(req.list.data).reshape((-1, 10))
        rospy.loginfo('received new task_list: ')
        rospy.loginfo(task_list)
        self.task_list = task_list.copy()  # copy a new duplicate, do not assign the reference

        self.path_plan_client.cancel_all_goals()

        # rospy.sleep(0.5)
        self.stop()

        return True

    def execute_task(self):
        if self.task_list is not None:
            self.no_more_task_warned = 0

            task_num = self.task_list.shape[0]
            task_list_cur = self.task_list[0, :].copy()
            rospy.logwarn('Executing task: ')
            rospy.logwarn(task_list_cur)

            if task_num == 1:
                self.task_list = None
            else:
                self.task_list = self.task_list[1:task_num, :].copy()

            if task_list_cur[0] == 0:  # stop mode, [0, ...]
                self.path_plan_client.cancel_all_goals()
                # publish all zero velocity cmd
                self.stop()
            elif task_list_cur[0] == 1:  # path plan mode, [1, x, y, theta, ...]
                params = task_list_cur[1:3].copy()
                path_np = self.path_plan_action(params)

                self.manipulation_action(path_np)
            elif task_list_cur[0] == 2:  # conveyor motion mode, [2, pos, ...]
                self.conveyor_motion_request(task_list_cur[1])
            else:
                rospy.logerr('unknown task code.')
        else:
            if not self.no_more_task_warned:
                rospy.logwarn('Task list empty now.')
                self.no_more_task_warned = 1

    # move_dir:
    # 1. move_base task: 0, stay still; 1, move forward; 2, move backward; 3, move left; 4, move right
    # 5, move forward no jump
    # 9, turn 180
    def path_plan_action(self, params):
        goal = cashmerebot.msg.path_planGoal()
        goal.params = params.tolist()

        self.path_plan_client.send_goal(goal)
        rospy.logwarn('path_plan_client: sent new goal (%f, %f)' % (goal.params[0], goal.params[1]))

        self.path_plan_client.wait_for_result()
        rospy.logwarn("path_plan_client: goal completed")

        path_list = self.path_plan_client.get_result()
        path_np = np.array(path_list.path).reshape((6,-1), order='F')

        return path_np

    def manipulation_action(self, path_np):
        goal = cashmerebot.msg.manipulationGoal()
        goal.path = path_np.T.reshape((-1,)).tolist()

        self.manipulation_client.send_goal(goal)
        rospy.logwarn('manipulation_client: sent new path')

        self.manipulation_client.wait_for_result()
        rospy.logwarn("manipulation_client: goal completed")

    def stop(self):
        pass

    def conveyor_motion_request(self, pos):
        msg = Float32MultiArray()
        msg.data = [pos]

        rospy.loginfo('send ConvCmd request: ')
        resp = self.conv_cmd_request(msg)
        rospy.loginfo('response is: %s' % (resp))


def main(args):
    rospy.init_node('task_manager_node', anonymous=True)
    tm = TaskManager()
    try:
        while not rospy.is_shutdown():
            tm.execute_task()
            rospy.sleep(0.2)
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
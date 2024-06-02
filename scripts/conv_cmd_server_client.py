#!/usr/bin/env python

from __future__ import print_function

import sys

import numpy
import rospy
from cashmerebot.srv import *
import numpy as np
from std_msgs.msg import Float32MultiArray
import math

pi = math.pi

# task1 = numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
task1 = numpy.array([[1]])


def send_task_list(tasks, request, msg):
    cmd_list = tasks.copy()

    cmd_list_flatten = cmd_list.reshape(1,)
    cmd_list_flatten_list = cmd_list_flatten.tolist()

    msg.data = cmd_list_flatten_list

    rospy.loginfo('send ConvCmd request: ')
    resp = request(msg)
    rospy.loginfo('response is: %s' % (resp))


if __name__ == "__main__":
    rospy.init_node('conv_cmd_server_client_node', anonymous=True)
    print('starting client')
    rospy.wait_for_service('ConvCmd')
    print('service connected.')
    try:
        task_list_request = rospy.ServiceProxy('ConvCmd', ConvCmd)
        msg = Float32MultiArray()

        send_task_list(task1, task_list_request, msg)
        rospy.sleep(20)

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
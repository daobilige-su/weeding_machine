#! /usr/bin/env python

import numpy as np
import rospy
import sys
from std_msgs.msg import String, Float32MultiArray
from cashmerebot.srv import ConvCmd


class ConvCmdServer:
    def __init__(self):

        # self.task_sleep_rate = rospy.Rate(10)

        self.task_list_srv = rospy.Service('ConvCmd', ConvCmd, self.conv_cmd_callback)
        rospy.loginfo('ConvCmd service ready')

        self.conv_cmd_pub = rospy.Publisher('ConvCmd_msg', Float32MultiArray, queue_size=2)

    # send conveyor position cmd via ros msg
    def conv_cmd_callback(self, req):
        cmd_list = np.array(req.list.data).reshape((1,))
        rospy.loginfo('received new cmd: ')
        rospy.loginfo(cmd_list)

        msg = Float32MultiArray()
        msg.data = cmd_list.tolist()
        self.conv_cmd_pub.publish(msg)

        rospy.sleep(1.0)

        return True


def main(args):
    rospy.init_node('conv_cmd_server_node', anonymous=True)
    srv = ConvCmdServer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
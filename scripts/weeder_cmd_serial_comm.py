#! /usr/bin/env python

import rospy
import sys
import tf
import serial  # https://pyserial.readthedocs.io/en/latest/shortintro.html
from std_msgs.msg import Float32MultiArray


class CmdSender:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0')

        self.weeder_cmd_sub = rospy.Subscriber('/weeder_cmd', Float32MultiArray, self.weeder_cmd_cb)

    def __del__(self):
        self.ser.close()

    def weeder_cmd_cb(self, msg):
        weeder_cmd = msg.data[0]

        ser_msg = '0x02'
        self.ser.write(ser_msg)

def main(args):
    rospy.init_node('weeder_cmd_serial_comm_node', anonymous=True)
    cmd_sender = CmdSender()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

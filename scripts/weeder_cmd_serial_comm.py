#! /usr/bin/env python

import rospy
import sys
import tf
import serial  # https://pyserial.readthedocs.io/en/latest/shortintro.html
from std_msgs.msg import Float32MultiArray

test_cmd = 1

class CmdSender:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0')
        self.ser.flush()

        self.weeder_cmd_sub = rospy.Subscriber('/weeder_cmd', Float32MultiArray, self.weeder_cmd_cb)

    def __del__(self):
        self.ser.close()

    def weeder_cmd_cb(self, msg):
        weeder_cmd = msg.data[0]
        weeder_cmd_mm = int(weeder_cmd*1000.0)
        if weeder_cmd_mm<0:
            weeder_cmd_mm = abs(weeder_cmd_mm)+4096

        weeder_cmd_mm_hex = f'{weeder_cmd_mm:0>4X}'

        ser_msg = b'\x02'
        self.ser.write(ser_msg)

        ser_msg = b'\x01'
        self.ser.write(ser_msg)

        ser_msg = b'\x04'
        self.ser.write(ser_msg)

        ser_msg = bytearray.fromhex(weeder_cmd_mm_hex)
        self.ser.write(ser_msg)

        ser_msg = b'\xff\xff'
        self.ser.write(ser_msg)

        ser_msg = b'\x03'
        self.ser.write(ser_msg)

        # self.ser.write(b'\n')

    def test_cmd(self):
        self.ser.flush()
        while True:
            if self.ser.read() == b'\x03':
                break
        while True:
            x = self.ser.read(8).hex()
            rospy.logwarn('CMD: ' + x)

        return

def main(args):
    rospy.init_node('weeder_cmd_serial_comm_node', anonymous=True)
    cmd_sender = CmdSender()

    if test_cmd:
        cmd_sender.test_cmd()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

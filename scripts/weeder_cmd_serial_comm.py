#! /usr/bin/env python

import rospy
import sys
import tf
# import time
import serial  # https://pyserial.readthedocs.io/en/latest/shortintro.html
from std_msgs.msg import Float32MultiArray, String, Float32

test_cmd = 1


class CmdSender:
    def __init__(self):

        # serial to weeder control
        ser_port = '/dev/ttyUSB0'
        self.ser = serial.Serial()
        self.ser.port = ser_port
        self.ser.baudrate = 115200
        self.ser.bytesize = 8  # len
        self.ser.stopbits = 1  # stop
        self.ser.parity = "N"  # check
        self.ser.open()

        # HMI vars
        self.HMI_data = ''

        # weeder vars
        self.weeder_data = None

        # pubs and subs
        self.speed_pub = rospy.Publisher("/weeder_speed", Float32, queue_size=1)
        self.weeder_cmd_sub = rospy.Subscriber('/weeder_cmd', Float32MultiArray, self.weeder_cmd_cb)
        self.rs485_pub = rospy.Publisher("rs485_send", String, queue_size=1)
        self.color_sub = rospy.Subscriber("HMI_send_data", String, self.hmi_cmd_cb,queue_size=1, buff_size=52428800)
        
    def __del__(self):
        self.ser.close()

    # extract and store weeder cmd
    def weeder_cmd_cb(self, msg):
        weeder_cmd = msg.data[0]
        weeder_cmd_mm = int(weeder_cmd*1000.0)
        if weeder_cmd_mm<0:
            weeder_cmd_mm = abs(weeder_cmd_mm)+4096
        self.weeder_data = "Weeder_distance:"+str(weeder_cmd_mm)

    # store hmi related data
    def hmi_cmd_cb(self, msg):
        """
        timer callback function
        """
        self.HMI_data = msg

    #
    def handle_hardware(self):
        self.ser.flush()
        while True:
            try:
                # send weeder cmd, and receive weeder weeder_speed and position (absolute? or relative?)
                if self.weeder_data is not None:
                    # send weeder control
                    self.ser.write(self.weeder_data.encode())
                    # sleep for 0.1s for mic to send back feedback data
                    rospy.sleep(0.1)
                    # read weeder feedback
                    data = self.ser.readline()

                    # get the weeder_speed, and parse it to the var below.
                    feedback_data = data[13:].decode('GBK')  # 数据1为反馈速度，单位为分米/秒,数据2为反馈机具偏移，单位为毫米
                    speed, pos = self.extract_speed_and_pos(feedback_data)

                    # publish weeder status
                    weeder_speed_msg = Float32()
                    weeder_speed_msg.data = speed
                    self.speed_pub.publish(weeder_speed_msg)
                    rospy.loginfo('weeder反馈:%s', weeder_speed_msg.data)

                # handle monitor hmi
                if self.HMI_data.data != '':    # monitor
                    self.ser.write(self.HMI_data.data.encode())
                    self.HMI_data.data = ''
                    rospy.sleep(0.1)
                    data = self.ser.readline()
                    msg = String()
                    msg.data = data.decode('GBK')
                    self.rs485_pub.publish(msg)
                    rospy.loginfo('HMI反馈:%s', msg)

            except:
                pass

    # func for extracting weeder weeder_speed and position
    def extract_speed_and_pos(self, str):
        speed_dm_str, pos_str = str.split(',')

        speed = float(speed_dm_str)/10.0
        if pos_str[-1]=='\n':
            pos_str = pos_str[:-1]
        pos = float(pos_str)

        return speed, pos


def main(args):
    rospy.init_node('weeder_cmd_serial_comm_node', anonymous=True)

    cmd_sender = CmdSender()
    cmd_sender.handle_hardware()  # program loops here

    # rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

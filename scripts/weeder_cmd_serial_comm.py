#! /usr/bin/env python

import rospy
import sys
import tf
# import time
import serial  # https://pyserial.readthedocs.io/en/latest/shortintro.html
from std_msgs.msg import Float32MultiArray, String, Float32
import yaml

test_cmd = 1


class CmdSender:
    def __init__(self):
        # load key params
        params_filename = rospy.get_param('param_file')  # self.pkg_path + 'cfg/' + 'param.yaml'
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)
        self.mic_response_time = self.param['weeder']['mic_response_time']
        self.hw_action = self.param['weeder']['hardware_interaction']
        self.verbose = self.param['weeder']['verbose']

        # serial to weeder control
        if self.hw_action:
            self.ser = serial.Serial()
            self.ser.port = self.param['weeder']['ser_port']
            self.ser.baudrate = self.param['weeder']['baudrate']
            self.ser.bytesize = self.param['weeder']['bytesize']  # len
            self.ser.stopbits = self.param['weeder']['stopbits']  # stop
            self.ser.parity = self.param['weeder']['parity']  # check
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
        if self.hw_action:
            self.ser.close()

    # extract and store weeder cmd
    def weeder_cmd_cb(self, msg):
        weeder_cmd = msg.data[0]
        weeder_cmd_mm = int(weeder_cmd*1000.0)  # meter to millimeter
        if weeder_cmd_mm<0:  # handle negative value for unsigned integer
            weeder_cmd_mm = abs(weeder_cmd_mm)+4096
        self.weeder_data = "Weeder_distance:"+str(weeder_cmd_mm)

    # store hmi related data
    def hmi_cmd_cb(self, msg):
        # timer callback function
        self.HMI_data = msg

    # communicate with weeder microcontroller, send and receive data
    def handle_hardware(self):
        if self.hw_action:
            self.ser.flush()
        while not rospy.is_shutdown():
            try:
                # send weeder cmd, and receive weeder weeder_speed and position (absolute? or relative?)
                if self.weeder_data is not None:
                    if self.hw_action:
                        # send weeder control
                        self.ser.write(self.weeder_data.encode())
                    # sleep for 0.1s for mic to send back feedback data
                    rospy.sleep(self.mic_response_time)

                    feedback_data = str(self.param['weeder']['def_speed']*10.0) + ',0.0' # default speed (m/s to dm/s)
                    if self.hw_action:
                        # read weeder feedback
                        data = self.ser.readline()

                        # get the weeder_speed, and parse it to the var below.
                        feedback_data = data[13:].decode('GBK')  # 数据1为反馈速度，单位为分米/秒,数据2为反馈机具偏移，单位为毫米
                    speed, pos = self.extract_speed_and_pos(feedback_data)

                    # publish weeder status
                    weeder_speed_msg = Float32()
                    weeder_speed_msg.data = speed
                    self.speed_pub.publish(weeder_speed_msg)
                    if self.verbose:
                        rospy.loginfo('weeder反馈:%s', weeder_speed_msg.data)

                # handle monitor hmi
                if self.HMI_data.data != '':    # monitor
                    if self.hw_action:
                        self.ser.write(self.HMI_data.data.encode())
                    self.HMI_data.data = ''
                    rospy.sleep(self.mic_response_time)
                    data = ''
                    if self.hw_action:
                        data = self.ser.readline()
                    msg = String()
                    msg.data = data.decode('GBK')
                    self.rs485_pub.publish(msg)
                    if self.verbose:
                        rospy.loginfo('HMI反馈:%s', msg)

            except:
                pass

    # func for extracting weeder weeder_speed and position
    def extract_speed_and_pos(self, speed_comma_pos):
        speed_dm_str, pos_str = speed_comma_pos.split(',')

        speed = float(speed_dm_str)/10.0
        if pos_str[-1] == '\n':
            pos_str = pos_str[:-1]
        pos = float(pos_str)

        return speed, pos


def main(args):
    rospy.init_node('weeder_cmd_serial_comm_node', anonymous=True)

    cmd_sender = CmdSender()
    cmd_sender.handle_hardware()  # program loops here


if __name__ == '__main__':
    main(sys.argv)

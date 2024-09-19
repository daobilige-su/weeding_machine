#!/usr/bin/env python


import rospy
import sys
import tf
# import time
import serial  # https://pyserial.readthedocs.io/en/latest/shortintro.html
from std_msgs.msg import Float32MultiArray, String, Float32
import yaml
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
            self.ser.timeout = 0.1  # Non-blocking read
            self.ser.open()

        # HMI vars
        self.HMI_data = ''

        # weeder vars
        self.weeder_data = "Weeder_distance: 0"
        self.feedback_data = None

        # pubs and subs
        self.speed_pos_pub = rospy.Publisher("/weeder_speed_pos", Float32MultiArray, queue_size=1)  # TODO
        self.weeder_cmd_sub = rospy.Subscriber('/weeder_cmd', Float32MultiArray, self.weeder_cmd_cb)
        self.weeder_sim_status_sub = rospy.Subscriber('/weeder_sim_status', Float32MultiArray, self.weeder_sim_status_cb)  # TODO
        self.rs485_pub = rospy.Publisher("rs485_send", String, queue_size=1)
        self.color_sub = rospy.Subscriber("HMI_send_data", String, self.hmi_cmd_cb,queue_size=1, buff_size=52428800)
        self.default_weeder_speed = self.param['weeder']['def_speed']  # TODO
        self.default_weeder_pos = self.param['weeder']['def_pos']  # TODO

        # Timers for handling hardware communication
        self.send_timer = rospy.Timer(rospy.Duration(0.1), self.send_data)  # Set timer to 0.1 seconds for sending
        self.receive_timer = rospy.Timer(rospy.Duration(0.1),
                                         self.receive_data)  # Set timer to 0.1 seconds for receiving

        # logging
        self.log_on = self.param['log']['enable']
        self.log_msg_pub = rospy.Publisher('/log_msg', String, queue_size=2)

    def __del__(self):
        if self.hw_action:
            self.ser.close()

    # extract and store weeder cmd
    def weeder_cmd_cb(self, msg):
        weeder_cmd = msg.data[0]
        rospy.loginfo(weeder_cmd)
        weeder_cmd = weeder_cmd
        weeder_cmd_mm = int(weeder_cmd * 1000.0)  # meter to millimeter
        if weeder_cmd_mm < 0:  # handle negative value for unsigned integer
            weeder_cmd_mm = abs(weeder_cmd_mm) + 4096
        self.weeder_data = "Weeder_distance:" + str(weeder_cmd_mm)

    # store hmi related data
    def hmi_cmd_cb(self, msg):
        self.HMI_data = msg

    # send data to hardware
    def send_data(self, event):
        if self.hw_action:
            # if self.HMI_data != '':
            #     self.ser.write(self.HMI_data.data.encode())
            #     self.HMI_data = ''  # Clear after sending
            # elif self.weeder_data is not None:
            #     self.ser.write(self.weeder_data.encode())
            #     self.weeder_data = None  # Clear after sending
            if self.HMI_data != '':
                self.ser.write(self.HMI_data.data.encode())
                self.HMI_data = ''  # Clear after sending
            if self.weeder_data is not None:
                self.ser.write(self.weeder_data.encode())
                # self.weeder_data = None  # Clear after sending
                if self.log_on:
                    log_msg = String()
                    log_msg.data = str(rospy.get_time()) + ': [SERIAL] sent weeder cmd, ' + str(self.weeder_data)
                    self.log_msg_pub.publish(log_msg)
                    rospy.sleep(0.001)

    def weeder_sim_status_cb(self, msg):
        self.default_weeder_speed = msg.data[0]
        self.default_weeder_pos = msg.data[1]
        return

    # receive data from hardware
    def receive_data(self, event):
        if self.hw_action:
            try:
                data = self.ser.readline()
                # if not data:
                #     feedback_data = str(self.default_weeder_speed * 10.0) + ',' + str(self.default_weeder_pos * 1000.0)  # TODO. m/s to dm/s, m to mm
                if data:
                    feedback_data = data.decode('GBK')
                    print(feedback_data)
                    if feedback_data.startswith("Weeder_speed,"):
                        speed, pos = self.extract_speed_and_pos(feedback_data[13:])

                        if self.log_on:
                            log_msg = String()
                            log_msg.data = str(rospy.get_time()) + ': [SERIAL] received weeder speed=' + str(
                                speed) + ', pos=' + str(pos)
                            self.log_msg_pub.publish(log_msg)
                            rospy.sleep(0.001)

                        # publish weeder status
                        weeder_speed_pos_msg = Float32MultiArray()
                        weeder_speed_pos_msg.data = [speed, pos]
                        self.speed_pos_pub.publish(weeder_speed_pos_msg)

                    else:
                        msg = String()
                        msg.data = feedback_data
                        self.rs485_pub.publish(msg)
                        if self.verbose:
                            rospy.loginfo('HMI feedback:%s', msg.data)
            except Exception as e:
                rospy.logerr("Error receiving data: %s", str(e))

    # func for extracting weeder speed and position
    def extract_speed_and_pos(self, speed_comma_pos):
        speed_dm_str, pos_str, _ = speed_comma_pos.split(',')
        speed = float(speed_dm_str) / 10.0  # dm/s to m/s
        if pos_str[-1] == '\n':
            pos_str = pos_str[:-1]
        pos = float(pos_str)/1000.0  # mm to m

        return speed, pos


def main(args):
    rospy.init_node('weeder_cmd_serial_comm_node', anonymous=True)

    cmd_sender = CmdSender()
    rospy.spin()  # Keeps the program alive


if __name__ == '__main__':
    main(sys.argv)
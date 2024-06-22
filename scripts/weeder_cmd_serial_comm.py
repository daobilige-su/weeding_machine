#! /usr/bin/env python

import rospy
import sys
import tf
import time
import serial  # https://pyserial.readthedocs.io/en/latest/shortintro.html
from std_msgs.msg import Float32MultiArray, String, Float32

test_cmd = 1

class CmdSender:
    def __init__(self):
        ser_port = '/dev/ttyUSB0'
        self.ser = serial.Serial()
        self.ser.port = ser_port
        self.ser.baudrate = 115200
        self.ser.bytesize = 8  # 设置数据位
        self.ser.stopbits = 1  # 设置停止位
        self.ser.parity = "N"  # 设置校验位
        self.ser.open()

        self.speed_pub = rospy.Publisher("/weeder_speed", Float32, queue_size=1)
        self.weeder_cmd_sub = rospy.Subscriber('/weeder_cmd', Float32MultiArray, self.weeder_cmd_cb)
        
        self.rs485_pub = rospy.Publisher("rs485_send", String, queue_size=1)
        self.color_sub = rospy.Subscriber("HMI_send_data", String, self.HMI_cmd_cb,
                                          queue_size=1, buff_size=52428800)
        
    def __del__(self):
        self.ser.close()

    def weeder_cmd_cb(self, msg):
        weeder_cmd = msg.data[0]
        weeder_cmd_mm = int(weeder_cmd*1000.0)
        if weeder_cmd_mm<0:
            weeder_cmd_mm = abs(weeder_cmd_mm)+4096

        # weeder_cmd_mm_hex = f'{weeder_cmd_mm:0>4X}'
        
        self.weeder_data = "Weeder_distance:"+str(weeder_cmd_mm)

    def HMI_cmd_cb(self,msg_back):
        """
        定时器回调函数
        """
        self.HMI_data = msg_back

    def test_cmd(self):
        self.ser.flush()
        while True:
            try:
                if self.HMI_data.data != '':    #显示器
                    self.ser.write(self.HMI_data.data.encode())
                    self.HMI_data.data = ''
                    time.sleep(0.1)
                    data = self.ser.readline()
                    msg = String()
                    msg.data = data.decode('GBK')
                    self.rs485_pub.publish(msg)
                    rospy.loginfo('HMI反馈:%s',msg)
                elif self.weeder_data != '':
                    self.ser.write(self.weeder_data.encode())
                    self.weeder_data = ''
                    time.sleep(0.1)
                    data = self.ser.readline()
                    msg = Float32()
                    feedback_data =  data[13:].decode('GBK')     #数据1为反馈速度，单位为分米/秒,数据2为反馈机具偏移，单位为毫米
                    # TODO, get the speed, and parse it to the var below.

                    msg.data = 0.5 # TODO, must change it to the weeder speed.
                    self.speed_pub.publish(msg)     # str: "speed,offset"
                    rospy.loginfo('weeder反馈:%s',msg.data)
            except:
                pass
                        
            

def main(args):
    rospy.init_node('weeder_cmd_serial_comm_node', anonymous=True)
    cmd_sender = CmdSender()

    if test_cmd:
        cmd_sender.test_cmd()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

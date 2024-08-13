#! /usr/bin/env python
import usb.core
import usb.util
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# 初始化ROS节点
rospy.init_node('camera_publisher')

# 创建Image消息的发布者
image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

# 创建一个用于转换OpenCV图像和ROS图像消息的桥接器
bridge = CvBridge()

# 相机的PID和VID
# 这里替换为您相机的VID和PID
CAMERA_PID = 0x0002

# 寻找对应VID和PID的设备
device = usb.core.find(idProduct=CAMERA_PID)

if device is None:
    raise ValueError('Camera not found')
else:
    rospy.loginfo(device)
    print('Camera found')



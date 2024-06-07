#! /usr/bin/env python
import numpy as np
import rospy
import sys
import tf
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import ros_numpy
from transform_tools import *
from numpy import matlib

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import filters

a = np.ones((24, 32))
u = np.tile(np.array([range(32)]), (24, 1))
v = np.tile(np.array([range(24)]).T, (1, 32))

u_flat = u.reshape((-1,))
v_flat = v.reshape((-1,))

uv = np.block([[u_flat], [v_flat]])

# K = np.array([159.99999300617776, 0.0, 160.0, 0.0, 159.99999300617776, 120.0, 0.0, 0.0, 1.0])
K = np.array([15.999999300617776, 0.0, 16.00, 0.0, 15.999999300617776, 12.00, 0.0, 0.0, 1.0]).reshape((3, 3))

K = np.array([[0]])


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
from mpl_toolkits import mplot3d
from skimage import filters

u = np.tile(np.array([range(320)]), (240, 1))
v = np.tile(np.array([range(240)]).T, (1, 320))

u_flat = u.reshape((-1,))
v_flat = v.reshape((-1,))

uv = np.block([[u_flat], [v_flat]])

# K = np.array([159.99999300617776, 0.0, 160.0, 0.0, 159.99999300617776, 120.0, 0.0, 0.0, 1.0])
K = np.array([159.99999300617776, 0.0, 160.00, 0.0, 159.99999300617776, 120.00, 0.0, 0.0, 1.0]).reshape((3, 3))

Rt_mc = transform_trans_ypr_to_matrix(np.array([0, 0, 1.2, np.deg2rad(-90), 0, np.deg2rad(-90-45)]))
Rt_cm = np.linalg.pinv(Rt_mc)

KRt = K @ Rt_cm[0:3, 0:4]  # 3x4

k11 = KRt[0, 0]
k12 = KRt[0, 1]
k13 = KRt[0, 2]
k14 = KRt[0, 3]
k21 = KRt[1, 0]
k22 = KRt[1, 1]
k23 = KRt[1, 2]
k24 = KRt[1, 3]
k31 = KRt[2, 0]
k32 = KRt[2, 1]
k33 = KRt[2, 2]
k34 = KRt[2, 3]

XY = np.divide(np.array([[k22*k34-k24*k32, k14*k32-k12*k34, k12*k24-k14*k22], [k24*k31-k21*k34, k11*k34-k14*k31, k14*k21-k11*k24]]) \
               @ np.block([[uv], [np.ones((1, uv.shape[1]))]]), \
               np.array([[k21*k32-k22*k31, k12*k31-k11*k32, k11*k22-k12*k21], [k21*k32-k22*k31, k12*k31-k11*k32, k11*k22-k12*k21]]) \
               @ np.block([[uv], [np.ones((1, uv.shape[1]))]]))
Z = np.zeros((1, uv.shape[1]))

P = np.block([[XY], [Z]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(P[0, :], P[1, :], P[2, :])
plt.show()

pass
# XY =

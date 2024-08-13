#! /usr/bin/env python
import numpy as np
import rospy
import sys
import tf
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float32
import ros_numpy
from transform_tools import *
from rospy.exceptions import ROSException

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch

import rospkg
import yaml

class cam_ext_tuner():
    def __init__(self):
        # KEY PARAMS
        self.cam_topic = '/camera/image_raw'
        self.K = np.array([226.2, 0.0, 172.5, 0.0, 225.7, 109.4, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.T_t_ypr = np.array([0, 0, -0.74, np.deg2rad(-2), 0, np.deg2rad(54)])
        self.theta_range = np.deg2rad(range(54, 90, 2))

        # locate ros pkg
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'

        # load key params
        params_filename = self.pkg_path + 'cfg/' + 'param_offline.yaml'
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)
        self.verbose = self.param['lane_det']['verbose']
        self.img_size = np.array([self.param['cam']['height'], self.param['cam']['width']])
        self.bird_pixel_size = self.param['lane_det']['bird_pixel_size']

        self.bridge = CvBridge()

    def tune_theta(self):
        # obtain camera images
        rospy.logwarn('waiting 100s for camera image ...')
        try:
            img_msg = rospy.wait_for_message(self.cam_topic, Image, 100)
        except ROSException as e:
            rospy.logwarn('image not received after 100s.')
            rospy.logwarn(e)
            return False
        rospy.logwarn('camera image received. ')

        cv_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(img_msg, "bgr8"))

        for theta in self.theta_range:
            T_t_ypr = self.T_t_ypr.copy()
            T_t_ypr[5] = theta
            rospy.loginfo('T: ')
            rospy.loginfo(T_t_ypr)
            T = transform_trans_ypr_to_matrix(np.array(T_t_ypr).reshape((6, 1)))
            bird_img_size, H = self.compute_homography(self.K, T)

            bird_eye_view_raw = cv2.warpPerspective(cv_img, H,
                                                    (int(bird_img_size[0]), int(bird_img_size[1])),
                                                    flags=cv2.INTER_NEAREST)

            mpl.use('TkAgg')
            plt.imshow(bird_eye_view_raw)
            # plt.show() # blocking mode, need to close the figure window
            plt.pause(0.1)

            rospy.sleep(2.0)

    def compute_homography(self, K, T):
        # extract four corners of the img, e.g. their u,v coords
        uv = np.array(
            [[0, self.img_size[1] - 1, self.img_size[1] - 1, 0], [0, 0, self.img_size[0] - 1, self.img_size[0] - 1]])

        Rt_mc = T  # T of camera in map
        Rt_cm = np.linalg.pinv(Rt_mc)  # T of map in camera

        KRt = K @ Rt_cm[0:3, 0:4]  # K * T, 3x4

        # the following code computes projection of uv points on the ground,
        # i.e. solution of rays intersecting with XY planes (Z=0)
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

        XY = np.divide(np.array([[k22 * k34 - k24 * k32, k14 * k32 - k12 * k34, k12 * k24 - k14 * k22],
                                 [k24 * k31 - k21 * k34, k11 * k34 - k14 * k31, k14 * k21 - k11 * k24]]) \
                       @ np.block([[uv], [np.ones((1, uv.shape[1]))]]), \
                       np.array([[k21 * k32 - k22 * k31, k12 * k31 - k11 * k32, k11 * k22 - k12 * k21],
                                 [k21 * k32 - k22 * k31, k12 * k31 - k11 * k32, k11 * k22 - k12 * k21]]) \
                       @ np.block([[uv], [np.ones((1, uv.shape[1]))]]))

        # prepare input image's u,v pairs of four corners
        inputpts = np.float32([[uv[0, 0], uv[1, 0]], [uv[0, 1], uv[1, 1]], [uv[0, 2], uv[1, 2]], [uv[0, 3], uv[1, 3]]])
        # prepare output image (bird eye view)'s u,v pairs of four corners
        outputpts = np.float32([[XY[0, 0], XY[1, 0]], [XY[0, 1], XY[1, 1]], [XY[0, 2], XY[1, 2]], [XY[0, 3], XY[1, 3]]])
        # make upper left point to be (0,0)
        outputpts[:, 1] = outputpts[:, 1] - np.min(outputpts[:, 1])
        outputpts[:, 0] = outputpts[:, 0] - np.min(outputpts[:, 0])
        # change meter to pixel
        outputpts = np.round(outputpts / self.bird_pixel_size)

        # compute max pixel coords in (u,v)
        bird_img_size = np.max(outputpts, axis=0)

        # compute homography matrix
        H = cv2.getPerspectiveTransform(inputpts, outputpts)

        # return vars
        return bird_img_size, H

    def resize_img_keep_scale(self, img):
        # compute aspect ratio
        img_ar = float(img.shape[0])/float(img.shape[1])
        img_size = [img.shape[0], img.shape[1]]
        tar_ar = float(self.img_size[0])/float(self.img_size[1])
        tar_size = [self.img_size[0], self.img_size[1]]

        # crop img
        if img_ar>tar_ar:  # crop vertically
            img_crop = img[max(int(float(img_size[0])/2.0 - float(img_size[1])*tar_ar/2.0), 0) \
                : min(int(float(img_size[0])/2.0 + float(img_size[1])*tar_ar/2.0), img_size[0]), :]
        else:  # crop horizontally
            img_crop = img[:, max(int(float(img_size[1])/2.0 - (float(img_size[0])/tar_ar)/2.0), 0) \
                : min(int(float(img_size[1])/2.0 + (float(img_size[0])/tar_ar)/2.0), img_size[1])]

        img_resize = cv2.resize(img_crop, (self.img_size[1], self.img_size[0]))

        return img_resize


def main(args):
    rospy.init_node('tune_cam_extrinsic_node', anonymous=True)
    tuner = cam_ext_tuner()

    # tune theta
    tuner.tune_theta()


if __name__ == '__main__':
    main(sys.argv)
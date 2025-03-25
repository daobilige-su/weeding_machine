#! /usr/bin/env python
import numpy as np
import rospy
import sys

import scipy.ndimage
import tf
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float32, String

import ros_numpy
from transform_tools import *
from rospy.exceptions import ROSException

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import multivariate_normal
from scipy import signal

# import torch

import rospkg
import yaml

import time
import ctypes

mpl.use('TkAgg')
# plt.imshow(image_seg_bn, cmap='gray')
# plt.show()

class intra_row:
    def __init__(self):
        # locate ros pkg
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'

        # load key params
        params_filename = rospy.get_param('param_file')  # self.pkg_path + 'cfg/' + 'param.yaml'
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)
        self.verbose = self.param['verbose']
        # self.img_size = np.array([self.param['cam']['height'], self.param['cam']['width']])
        self.cam_img_size = np.array(self.param['cam']['img_size'], dtype=int)

        # plant detection using yolo5 / ExG++
        self.segment_mode = self.param['intra_row']['segment_mode']
        self.yolo5_plant_id = self.param['intra_row']['yolo_plant_id']
        self.yolo5_model_path = self.pkg_path + 'weights/' + self.param['lane_det']['yolo_model']
        if self.segment_mode == 1:
            import torch
            self.model = torch.hub.load(self.pkg_path + 'yolov5', 'custom', path=self.yolo5_model_path, source='local')
            rospy.loginfo('using yolo5 based plant segmentation')
        elif self.segment_mode == 2:
            self.model = None
            rospy.loginfo('using ExG based plant segmentation')
        else:
            rospy.logerr('unknown segmentation mode, returning...')
            return

        # cam vars, pubs and subs
        self.img1 = np.zeros((self.cam_img_size[0], self.cam_img_size[1], 3), dtype=np.uint8)
        self.img2 = np.zeros((self.cam_img_size[0], self.cam_img_size[1], 3), dtype=np.uint8)
        self.img3 = np.zeros((self.cam_img_size[0], self.cam_img_size[1], 3), dtype=np.uint8)
        self.img4 = np.zeros((self.cam_img_size[0], self.cam_img_size[1], 3), dtype=np.uint8)
        self.img5 = np.zeros((self.cam_img_size[0], self.cam_img_size[1], 3), dtype=np.uint8)
        self.img6 = np.zeros((self.cam_img_size[0], self.cam_img_size[1], 3), dtype=np.uint8)
        self.cam1_topic = self.param['cam']['cam1_topic']
        self.cam2_topic = self.param['cam']['cam2_topic']
        self.cam3_topic = self.param['cam']['cam3_topic']
        self.cam4_topic = self.param['cam']['cam4_topic']
        self.cam5_topic = self.param['cam']['cam5_topic']
        self.cam6_topic = self.param['cam']['cam6_topic']

        self.bridge = CvBridge()

        self.img1_sub = rospy.Subscriber(self.cam1_topic, Image, self.img1_cb, queue_size=1)
        self.img2_sub = rospy.Subscriber(self.cam2_topic, Image, self.img2_cb, queue_size=1)
        self.img3_sub = rospy.Subscriber(self.cam3_topic, Image, self.img3_cb, queue_size=1)
        self.img4_sub = rospy.Subscriber(self.cam4_topic, Image, self.img4_cb, queue_size=1)
        self.img5_sub = rospy.Subscriber(self.cam5_topic, Image, self.img5_cb, queue_size=1)
        self.img6_sub = rospy.Subscriber(self.cam6_topic, Image, self.img6_cb, queue_size=1)

        self.img_whole_pub = rospy.Publisher('/img_whole/image_raw', Image, queue_size=2)


        # self.img_whole = np.zeros((self.cam_img_size[0]*3, self.cam_img_size[1]*2, 3), dtype=np.uint8)
        self.img_whole = np.zeros((270 * 3 - 30, 350 * 2 - 30, 3), dtype=np.uint8)

        self.weeder_speed = 0.1
        # self.plant_loc = -1*np.ones((12, 20))
        self.plant_loc = np.zeros((12, 200))  # -50~150
        self.plant_loc_offset = 12
        self.weeder_speed_scaling = 1.0

        self.img_cb_pre_t = None

        # mpl.use('TkAgg')
        # self.fig, self.axs = plt.subplots(2)
        # plt.show()


        # self.cam_to_use = self.param['cam']['cam_to_use']
        # self.cam_sel_time_interval = self.param['cam']['cam_sel_time_interval']
        # self.choose_cam_pre_time = rospy.get_time()
        # if self.cam_to_use == -1:
        #     if not self.select_cam():
        #         rospy.logwarn('camera auto selection failed.')
        #         return
        # # self.left_img_sub = rospy.Subscriber(self.param['cam']['left']['topic'], Image, self.left_img_cb, queue_size=1)
        # # self.right_img_sub = rospy.Subscriber(self.param['cam']['right']['topic'], Image, self.right_img_cb, queue_size=1)
        # self.proc_img_pub = rospy.Publisher('/proc_img/image_raw', Image, queue_size=2)
        self.seg_img_pub = rospy.Publisher('/seg_img/image_raw', Image, queue_size=2)
        # # self.det_img_left_pub = rospy.Publisher('/det_img/image_raw_left', Image, queue_size=2)
        # # self.det_img_right_pub = rospy.Publisher('/det_img/image_raw_right', Image, queue_size=2)
        self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=2)
        # self.seg_line_pub = rospy.Publisher('/seg_line/image_raw', Image, queue_size=2)
        # # self.line_rgb_left_pub = rospy.Publisher('/line_rgb/image_raw_left', Image, queue_size=2)
        # # self.line_rgb_right_pub = rospy.Publisher('/line_rgb/image_raw_right', Image, queue_size=2)
        # self.rgb_line_pub = rospy.Publisher('/rgb_line/image_raw', Image, queue_size=2)
        # # get camera intrinsic
        # self.left_cam_K = np.array(self.param['cam']['left']['K']).reshape((3, 3))
        # self.right_cam_K = np.array(self.param['cam']['right']['K']).reshape((3, 3))
        # # get camera extrinsic
        # self.left_cam_T = transform_trans_ypr_to_matrix(np.array(self.param['cam']['left']['T']).reshape((6, 1)))
        # self.right_cam_T = transform_trans_ypr_to_matrix(np.array(self.param['cam']['right']['T']).reshape((6, 1)))
        #
        # # lane det params
        # self.wrap_img_on = self.param['lane_det']['wrap_img_on']
        # self.wrap_img_size_u = self.param['lane_det']['wrap_img_size_u']
        # # wrap_param: [left lane bot pix u, right lane bot pix u, left lane top pix u, right lane top pix u, bot v, top v, weeder bot u]
        # self.wrap_param_left_cam = np.array(self.param['lane_det']['wrap_param_left_cam']).reshape((-1,))
        # self.wrap_param_right_cam = np.array(self.param['lane_det']['wrap_param_right_cam']).reshape((-1,))
        # self.wrap_H_left_cam, self.wrap_H_right_cam = self.compute_wrap_H(self.wrap_param_left_cam, self.wrap_param_right_cam)
        # self.wrap_param = None  # dynamically change according to left and right cam selection
        # self.wrap_img_size = None
        # self.wrap_H = None  # dynamically change according to left and right cam selection
        # self.resize_proportion = None # proportion of resize operation in image wrapping
        #
        # self.cfunc_handle = ctypes.CDLL(self.pkg_path + "libGrayScaleHoughLine.so") # cpp function call
        # self.lane_det_track_img = None
        # self.lane_det_track_u = None
        # self.lane_det_lines = None
        #
        # # weeder pub, sub and vars
        # self.weeder_speed = self.param['weeder']['def_speed']
        # self.weeder_pos = self.param['weeder']['def_pos']
        # self.weeder_status = self.param['weeder']['def_status']
        self.weeder_control_pub = rospy.Publisher('/weeder_cmd', Float32MultiArray, queue_size=2)
        # self.weeder_speed_pos_sub = rospy.Subscriber('/weeder_speed_pos', Float32MultiArray, self.weeder_speed_pos_cb)
        # self.weeder_cmd = 0
        # if self.param['weeder']['cmd_dist'] == -1:
        #     self.weeder_cmd_buff = np.zeros((int(self.param['weeder']['cam_weeder_dist'] / self.param['weeder']['cmd_dist_def']), ))
        # else:
        #     self.weeder_cmd_buff = np.zeros((int(self.param['weeder']['cam_weeder_dist']/self.param['weeder']['cmd_dist']),))
        # self.ctl_time_pre = rospy.get_time()
        # self.weeder_cmd_delay = self.param['weeder']['weeder_cmd_delay']
        # self.lane_y_offset = 0.0
        #
        # logging
        self.log_on = self.param['log']['enable']
        self.log_msg_pub = rospy.Publisher('/log_msg', String, queue_size=2)
        #
        # # # TODO: TEST
        # # self.test_t = rospy.get_time()

        return

    # sub for cam1 - cam6 img, store it in member var and call line detection if left cam is selected.
    def img1_cb(self, msg):
        # store var
        try:
            self.img1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

    def img2_cb(self, msg):
        # store var
        try:
            self.img2 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

    def img3_cb(self, msg):
        # store var
        try:
            self.img3 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

    def img4_cb(self, msg):
        # store var
        try:
            self.img4 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

    def img5_cb(self, msg):
        # store var
        try:
            self.img5 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

    def img6_cb(self, msg):
        # store var
        try:
            self.img6 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return


    # sub for weeder weeder speed and pos, simply store it in member var
    def weeder_speed_pos_cb(self, msg):
        self.weeder_speed = msg.data[0]
        self.weeder_pos = msg.data[1]
        self.weeder_status = msg.data[2]


    # the main loop: detect line and send back the result
    def img_cb(self):
        cur_t = rospy.get_time()

        if self.img_cb_pre_t is not None:
            delta_t = cur_t - self.img_cb_pre_t
            shift = int((self.weeder_speed * self.weeder_speed_scaling * delta_t)*100)

            if shift<200:
                self.plant_loc[:, shift:200] = self.plant_loc[:, 0:200 - shift]
            else:
                rospy.logwarn('shift too large.')
                self.plant_loc = np.zeros((12, 200))

        self.img_cb_pre_t = cur_t

        log_msg = String()
        t0 = rospy.get_time()
        if self.log_on:
            log_msg.data = str(rospy.get_time()) + ': start processing a new image.'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)

        # (0) choose camera in each cam_sel_time_interval time
        # self.img_whole[self.cam_img_size[0] * 0:self.cam_img_size[0] * 1,
        #     self.cam_img_size[1] * 0:self.cam_img_size[1] * 1, :] = self.img1
        # self.img_whole[self.cam_img_size[0] * 0:self.cam_img_size[0] * 1,
        #     self.cam_img_size[1] * 1:self.cam_img_size[1] * 2, :] = self.img2
        # self.img_whole[self.cam_img_size[0] * 1:self.cam_img_size[0] * 2,
        #     self.cam_img_size[1] * 0:self.cam_img_size[1] * 1, :] = self.img3
        # self.img_whole[self.cam_img_size[0] * 1:self.cam_img_size[0] * 2,
        #     self.cam_img_size[1] * 1:self.cam_img_size[1] * 2, :] = self.img4
        # self.img_whole[self.cam_img_size[0] * 2:self.cam_img_size[0] * 3,
        #     self.cam_img_size[1] * 0:self.cam_img_size[1] * 1, :] = self.img5
        # self.img_whole[self.cam_img_size[0] * 2:self.cam_img_size[0] * 3,
        #     self.cam_img_size[1] * 1:self.cam_img_size[1] * 2, :] = self.img6
        self.img_whole[270 * 0:270 * 0 + self.cam_img_size[0],
            350 * 0:350 * 0 + self.cam_img_size[1], :] = self.img1
        self.img_whole[270 * 0:270 * 0 + self.cam_img_size[0],
            350 * 1:350 * 1 + self.cam_img_size[1], :] = self.img2
        self.img_whole[270 * 1:270 * 1 + self.cam_img_size[0],
            350 * 0:350 * 0 + self.cam_img_size[1], :] = self.img3
        self.img_whole[270 * 1:270 * 1 + self.cam_img_size[0],
            350 * 1:350 * 1 + self.cam_img_size[1], :] = self.img4
        self.img_whole[270 * 2:270 * 2 + self.cam_img_size[0],
            350 * 0:350 * 0 + self.cam_img_size[1], :] = self.img5
        self.img_whole[270 * 2:270 * 2 + self.cam_img_size[0],
            350 * 1:350 * 1 + self.cam_img_size[1], :] = self.img6

        # cv2.line(self.img_whole, (0, self.cam_img_size[0]), (self.cam_img_size[1]*2, self.cam_img_size[0]), (0, 0, 255),
        #          1, cv2.LINE_AA)
        # cv2.line(self.img_whole, (0, self.cam_img_size[0]*2), (self.cam_img_size[1] * 2, self.cam_img_size[0]*2),
        #          (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.line(self.img_whole, (self.cam_img_size[1], 0), (self.cam_img_size[1], self.cam_img_size[0]*3),
        #          (0, 0, 255), 1, cv2.LINE_AA)

        # publish image to be processed
        try:
            ros_image = self.bridge.cv2_to_imgmsg(self.img_whole, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        self.img_whole_pub.publish(ros_image)

        #
        #
        # choose_cam_cur_time = rospy.get_time()
        # if choose_cam_cur_time - self.choose_cam_pre_time >= self.cam_sel_time_interval:
        #     self.select_cam()
        #     self.lane_det_track_img = None  # re-initialize tracking of the lane det result
        #     self.choose_cam_pre_time = rospy.get_time()
        #
        #     t01 = rospy.get_time()
        #     if self.log_on:
        #         log_msg.data = str(t01) + ': finished camera selection process.'
        #         self.log_msg_pub.publish(log_msg)
        #         rospy.sleep(0.001)
        #
        # # # TODO: TEST
        # # t = rospy.get_time()
        # # if t - self.test_t > 5:
        # #     if self.cam_to_use == 0:
        # #         self.cam_to_use = 1
        # #     else:
        # #         self.cam_to_use = 0
        # #     self.lane_det_track_img = None
        # #     self.test_t = t
        # #     rospy.logwarn('changed cam to cam: ' + str(self.cam_to_use))
        #
        # # select left or right camera image, and their corresponding params for lane detection
        # if self.cam_to_use == 0:
        #     cv_image = self.left_img
        #     self.wrap_param = self.wrap_param_left_cam
        #     self.wrap_H = self.wrap_H_left_cam
        #     if self.log_on:
        #         log_msg.data = str(rospy.get_time()) + ': using left camera image.'
        #         self.log_msg_pub.publish(log_msg)
        #         rospy.sleep(0.001)
        # elif self.cam_to_use == 1:
        #     cv_image = self.right_img
        #     self.wrap_param = self.wrap_param_right_cam
        #     self.wrap_H = self.wrap_H_right_cam
        #     if self.log_on:
        #         log_msg.data = str(rospy.get_time()) + ': using right camera image.'
        #         self.log_msg_pub.publish(log_msg)
        #         rospy.sleep(0.001)
        # else:
        #     rospy.logwarn('lane_det: unknown camera selection.')
        #     return
        # self.wrap_img_size = np.array([int((self.wrap_param[4] - self.wrap_param[5] + 1) *
        #                                    (self.wrap_img_size_u / (2 * (self.wrap_param[1] - self.wrap_param[0])))), self.wrap_img_size_u])
        # self.resize_proportion = (self.wrap_img_size_u / ((self.wrap_param[1] - self.wrap_param[0]) * 2.0))
        #
        # if cv_image is None:
        #     if self.log_on:
        #         log_msg.data = str(rospy.get_time()) + ': camera image not ready, skipping.'
        #         self.log_msg_pub.publish(log_msg)
        #         rospy.sleep(0.001)
        #     rospy.sleep(0.5)
        #     return
        # else:
        #     # publish image to be processed
        #     try:
        #         ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        #     except CvBridgeError as e:
        #         print(e)
        #         return
        #     self.proc_img_pub.publish(ros_image)
        #
        # t02 = rospy.get_time()
        # if self.verbose:
        #     rospy.loginfo('time consumption until preparing image = %f' % (t02 - t0))
        # if self.log_on:
        #     log_msg.data = str(rospy.get_time()) + ': finished preparing image.'
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #     log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t02 - t0)
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #
        # (1) segment plants in the image
        # segment plants with yolov5 / ExG
        if self.segment_mode == 1:
            plant_seg_bn, det_image, plant_seg_gaus, det_ctr, det_bbox = self.segment_plants_yolo5(self.img_whole)
        elif self.segment_mode == 2:
            plant_seg_bn, det_image, plant_seg_gaus, det_ctr, det_bbox = self.segment_plants_exg(self.img_whole)
        else:
            rospy.logerr('unknown segmentation mode, returning...')
            return
        #
        # publish segmentation results
        plant_seg_pub = np.asarray((plant_seg_bn * 255).astype(np.uint8))
        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_pub, "mono8")
            det_ros_image = self.bridge.cv2_to_imgmsg(det_image,"bgr8")
        except CvBridgeError as e:
            print(e)
            return
        self.seg_img_pub.publish(ros_image)
        # if self.cam_to_use==0:
        #     self.det_img_left_pub.publish(det_image)
        # elif self.cam_to_use==1:
        #     self.det_img_right_pub.publish(det_image)
        self.det_img_pub.publish(det_ros_image)

        t1 = rospy.get_time()
        if self.verbose:
            if self.segment_mode == 1:
                rospy.loginfo('time consumption until yolo5 based image det and seg = %f' % (t1 - t0))
            elif self.segment_mode == 2:
                rospy.loginfo('time consumption until ExG based image det and seg = %f' % (t1 - t0))
        if self.log_on:
            if self.segment_mode == 1:
                log_msg.data = str(rospy.get_time()) + ': finished yolo5 based image det and seg.'
            elif self.segment_mode == 2:
                log_msg.data = str(rospy.get_time()) + ': finished ExG based image det and seg.'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t1 - t0)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)

        # (2) compute proj of det_ctr
        sub_seg_img1 = None
        sub_seg_img2 = None
        sub_seg_img3 = None
        sub_seg_img4 = None
        sub_seg_img5 = None
        sub_seg_img6 = None
        sub_det_ctr1 = None
        sub_det_ctr2 = None
        sub_det_ctr3 = None
        sub_det_ctr4 = None
        sub_det_ctr5 = None
        sub_det_ctr6 = None
        sub_det_bbox1 = None
        sub_det_bbox2 = None
        sub_det_bbox3 = None
        sub_det_bbox4 = None
        sub_det_bbox5 = None
        sub_det_bbox6 = None
        row_det_ctr = np.zeros((12, 10, 2))
        row_det_ctr_num = np.zeros((12, 1), dtype=int)

        # sub_seg_img1 = self.img_whole[0: self.cam_img_size[0], 0: self.cam_img_size[1]]
        # sub_seg_img2 = self.img_whole[0: self.cam_img_size[0], self.cam_img_size[1]: self.cam_img_size[1]*2]
        # sub_seg_img3 = self.img_whole[self.cam_img_size[0]: self.cam_img_size[0]*2, 0: self.cam_img_size[1]]
        # sub_seg_img4 = self.img_whole[self.cam_img_size[0]: self.cam_img_size[0]*2, self.cam_img_size[1]: self.cam_img_size[1]*2]
        # sub_seg_img5 = self.img_whole[self.cam_img_size[0]*2: self.cam_img_size[0]*3, 0: self.cam_img_size[1]]
        # sub_seg_img6 = self.img_whole[self.cam_img_size[0]*2: self.cam_img_size[0]*3, self.cam_img_size[1]: self.cam_img_size[1]*2]
        sub_seg_img1 = self.img_whole[0: self.cam_img_size[0], 0: self.cam_img_size[1]]
        sub_seg_img2 = self.img_whole[0: self.cam_img_size[0], 350: 350 + self.cam_img_size[1]]
        sub_seg_img3 = self.img_whole[270: 270 + self.cam_img_size[0], 0: self.cam_img_size[1]]
        sub_seg_img4 = self.img_whole[270: 270 + self.cam_img_size[0], 350: 350 + self.cam_img_size[1]]
        sub_seg_img5 = self.img_whole[270 * 2: 270*2 + self.cam_img_size[0], 0: self.cam_img_size[1]]
        sub_seg_img6 = self.img_whole[270 * 2: 270*2 + self.cam_img_size[0], 350: 350 + self.cam_img_size[1]]

        # det_ctr: [x, y], x is along shape[1], y is along shape[0]
        if det_ctr is not None:
            for n in range(det_ctr.shape[0]):
                ctr = det_ctr[n, :]
                bbox = det_bbox[n, :]
                if ctr[1]>=0 and ctr[1]<270:
                    if ctr[0]>=0 and ctr[0]<350:
                        if sub_det_ctr1 is None:
                            sub_det_ctr1 = np.array([ctr])
                            sub_det_bbox1 = np.array([bbox])
                        else:
                            sub_det_ctr1 = np.block([[sub_det_ctr1], [ctr]])
                            sub_det_bbox1 = np.block([[sub_det_bbox1], [bbox]])
                    elif ctr[0]>=350:
                        if sub_det_ctr2 is None:
                            sub_det_ctr2 = np.array([ctr])
                            sub_det_bbox2 = np.array([bbox])
                        else:
                            sub_det_ctr2 = np.block([[sub_det_ctr2], [ctr]])
                            sub_det_bbox2 = np.block([[sub_det_bbox2], [bbox]])
                    else:
                        rospy.logwarn('unknown center value in X.')
                elif ctr[1]>=270 and ctr[1]<270*2:
                    if ctr[0]>=0 and ctr[0]<350:
                        if sub_det_ctr3 is None:
                            sub_det_ctr3 = np.array([ctr])
                            sub_det_bbox3 = np.array([bbox])
                        else:
                            sub_det_ctr3 = np.block([[sub_det_ctr3], [ctr]])
                            sub_det_bbox3 = np.block([[sub_det_bbox3], [bbox]])
                    elif ctr[0]>=350:
                        if sub_det_ctr4 is None:
                            sub_det_ctr4 = np.array([ctr])
                            sub_det_bbox4 = np.array([bbox])
                        else:
                            sub_det_ctr4 = np.block([[sub_det_ctr4], [ctr]])
                            sub_det_bbox4 = np.block([[sub_det_bbox4], [bbox]])
                    else:
                        rospy.logwarn('unknown center value in X.')
                elif ctr[1]>=270*2 and ctr[1]<270*3:
                    if ctr[0]>=0 and ctr[0]<350:
                        if sub_det_ctr5 is None:
                            sub_det_ctr5 = np.array([ctr])
                            sub_det_bbox5 = np.array([bbox])
                        else:
                            sub_det_ctr5 = np.block([[sub_det_ctr5], [ctr]])
                            sub_det_bbox5 = np.block([[sub_det_bbox5], [bbox]])
                    elif ctr[0]>=350:
                        if sub_det_ctr6 is None:
                            sub_det_ctr6 = np.array([ctr])
                            sub_det_bbox6 = np.array([bbox])
                        else:
                            sub_det_ctr6 = np.block([[sub_det_ctr6], [ctr]])
                            sub_det_bbox6 = np.block([[sub_det_bbox6], [bbox]])
                    else:
                        rospy.logwarn('unknown center value in X.')
                else:
                    rospy.logwarn('unknown center value in Y.')

        det_image_rgb = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
        if sub_det_ctr1 is not None:
            for n in range(sub_det_ctr1.shape[0]):
                ctr = sub_det_ctr1[n, :].copy()
                sub_det_ctr1[n, :] = np.array([ctr[0] - 350 * 0, ctr[1] - 270 * 0])
                bbox = sub_det_bbox1[n, :].copy()
                sub_det_bbox1[n, :] = np.array([bbox[0] - 350 * 0, bbox[1] - 270 * 0, bbox[2] - 350 * 0, bbox[3] - 270 * 0])
                bbox = sub_det_bbox1[n, :]
                if bbox[1]>self.cam_img_size[0]*0.1 and bbox[3]<self.cam_img_size[0]*0.9:
                    ctr = sub_det_ctr1[n, :]
                    if ctr[0]<self.cam_img_size[1]/2:
                        row_det_ctr[0, row_det_ctr_num[0], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[0, row_det_ctr_num[0], 0]) + 350*0, int(row_det_ctr[0, row_det_ctr_num[0], 1]) + 270 * 0), 8, (0, 0, 62), -1)
                        row_det_ctr_num[0] = row_det_ctr_num[0] + 1
                    else:
                        row_det_ctr[1, row_det_ctr_num[1], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[1, row_det_ctr_num[1], 0]) + 350*0, int(row_det_ctr[1, row_det_ctr_num[1], 1])+ 270 * 0), 8, (0, 0, 125), -1)
                        row_det_ctr_num[1] = row_det_ctr_num[1] + 1
        if sub_det_ctr2 is not None:
            for n in range(sub_det_ctr2.shape[0]):
                ctr = sub_det_ctr2[n, :].copy()
                sub_det_ctr2[n, :] = np.array([ctr[0] - 350 * 1, ctr[1] - 270 * 0])
                bbox = sub_det_bbox2[n, :].copy()
                sub_det_bbox2[n, :] = np.array([bbox[0] - 350 * 1, bbox[1] - 270 * 0, bbox[2] - 350 * 1, bbox[3] - 270 * 0])
                bbox = sub_det_bbox2[n, :]
                if bbox[1] > self.cam_img_size[0] * 0.1 and bbox[3] < self.cam_img_size[0] * 0.9:
                    ctr = sub_det_ctr2[n, :]
                    if ctr[0]<self.cam_img_size[1]/2:
                        row_det_ctr[2, row_det_ctr_num[2], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[2, row_det_ctr_num[2], 0]) + 350*1, int(row_det_ctr[2, row_det_ctr_num[2], 1])+ 270 * 0), 8, (0, 0, 190), -1)
                        row_det_ctr_num[2] = row_det_ctr_num[2] + 1
                    else:
                        row_det_ctr[3, row_det_ctr_num[3], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[3, row_det_ctr_num[3], 0]) + 350*1, int(row_det_ctr[3, row_det_ctr_num[3], 1])+ 270 * 0), 8, (0, 0, 255), -1)
                        row_det_ctr_num[3] = row_det_ctr_num[3] + 1
        if sub_det_ctr3 is not None:
            for n in range(sub_det_ctr3.shape[0]):
                ctr = sub_det_ctr3[n, :].copy()
                sub_det_ctr3[n, :] = np.array([ctr[0] - 350 * 0, ctr[1] - 270 * 1])
                bbox = sub_det_bbox3[n, :].copy()
                sub_det_bbox3[n, :] = np.array([bbox[0] - 350 * 0, bbox[1] - 270 * 1, bbox[2] - 350 * 0, bbox[3] - 270 * 1])
                bbox = sub_det_bbox3[n, :]
                if bbox[1] > self.cam_img_size[0] * 0.1 and bbox[3] < self.cam_img_size[0] * 0.9:
                    ctr = sub_det_ctr3[n, :]
                    if ctr[0] < self.cam_img_size[1] / 2:
                        row_det_ctr[4, row_det_ctr_num[4], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[4, row_det_ctr_num[4], 0]) + 350*0, int(row_det_ctr[4, row_det_ctr_num[4], 1])+ 270 * 1), 8, (0, 62, 0), -1)
                        row_det_ctr_num[4] = row_det_ctr_num[4] + 1
                    else:
                        row_det_ctr[5, row_det_ctr_num[5], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[5, row_det_ctr_num[5], 0]) + 350*0, int(row_det_ctr[5, row_det_ctr_num[5], 1])+ 270 * 1), 8, (0, 125, 0), -1)
                        row_det_ctr_num[5] = row_det_ctr_num[5] + 1
        if sub_det_ctr4 is not None:
            for n in range(sub_det_ctr4.shape[0]):
                ctr = sub_det_ctr4[n, :].copy()
                sub_det_ctr4[n, :] = np.array([ctr[0] - 350 * 1, ctr[1] - 270 * 1])
                bbox = sub_det_bbox4[n, :].copy()
                sub_det_bbox4[n, :] = np.array([bbox[0] - 350 * 1, bbox[1] - 270 * 1, bbox[2] - 350 * 1, bbox[3] - 270 * 1])
                bbox = sub_det_bbox4[n, :]
                if bbox[1] > self.cam_img_size[0] * 0.1 and bbox[3] < self.cam_img_size[0] * 0.9:
                    ctr = sub_det_ctr4[n, :]
                    if ctr[0] < self.cam_img_size[1] / 2:
                        row_det_ctr[6, row_det_ctr_num[6], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[6, row_det_ctr_num[6], 0]) + 350*1, int(row_det_ctr[6, row_det_ctr_num[6], 1])+ 270 * 1), 8, (0, 190, 0), -1)
                        row_det_ctr_num[6] = row_det_ctr_num[6] + 1
                    else:
                        row_det_ctr[7, row_det_ctr_num[7], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[7, row_det_ctr_num[7], 0]) + 350*1, int(row_det_ctr[7, row_det_ctr_num[7], 1])+ 270 * 1), 8, (0, 255, 0), -1)
                        row_det_ctr_num[7] = row_det_ctr_num[7] + 1
        if sub_det_ctr5 is not None:
            for n in range(sub_det_ctr5.shape[0]):
                ctr = sub_det_ctr5[n, :].copy()
                sub_det_ctr5[n, :] = np.array([ctr[0] - 350 * 0, ctr[1] - 270 * 2])
                bbox = sub_det_bbox5[n, :].copy()
                sub_det_bbox5[n, :] = np.array([bbox[0] - 350 * 0, bbox[1] - 270 * 2, bbox[2] - 350 * 0, bbox[3] - 270 * 2])
                bbox = sub_det_bbox5[n, :]
                if bbox[1] > self.cam_img_size[0] * 0.1 and bbox[3] < self.cam_img_size[0] * 0.9:
                    ctr = sub_det_ctr5[n, :]
                    if ctr[0] < self.cam_img_size[1] / 2:
                        row_det_ctr[8, row_det_ctr_num[8], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[8, row_det_ctr_num[8], 0]) + 350*0, int(row_det_ctr[8, row_det_ctr_num[8], 1])+ 270 * 2), 8, (62, 0, 0), -1)
                        row_det_ctr_num[8] = row_det_ctr_num[8] + 1
                    else:
                        row_det_ctr[9, row_det_ctr_num[9], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[9, row_det_ctr_num[9], 0]) + 350*0, int(row_det_ctr[9, row_det_ctr_num[9], 1])+ 270 * 2), 8, (125, 0, 0), -1)
                        row_det_ctr_num[9] = row_det_ctr_num[9] + 1
        if sub_det_ctr6 is not None:
            for n in range(sub_det_ctr6.shape[0]):
                ctr = sub_det_ctr6[n, :].copy()
                sub_det_ctr6[n, :] = np.array([ctr[0] - 350 * 1, ctr[1] - 270 * 2])
                bbox = sub_det_bbox6[n, :].copy()
                sub_det_bbox6[n, :] = np.array([bbox[0] - 350 * 1, bbox[1] - 270 * 2, bbox[2] - 350 * 1, bbox[3] - 270 * 2])
                bbox = sub_det_bbox6[n, :]
                if bbox[1] > self.cam_img_size[0] * 0.1 and bbox[3] < self.cam_img_size[0] * 0.9:
                    ctr = sub_det_ctr6[n, :]
                    if ctr[0] < self.cam_img_size[1] / 2:
                        row_det_ctr[10, row_det_ctr_num[10], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[10, row_det_ctr_num[10], 0]) + 350*1, int(row_det_ctr[10, row_det_ctr_num[10], 1])+ 270 * 2), 8, (190, 0, 0), -1)
                        row_det_ctr_num[10] = row_det_ctr_num[10] + 1
                    else:
                        row_det_ctr[11, row_det_ctr_num[11], :] = ctr
                        cv2.circle(det_image_rgb, (int(row_det_ctr[11, row_det_ctr_num[11], 0]) + 350*1, int(row_det_ctr[11, row_det_ctr_num[11], 1])+ 270 * 2), 8, (255, 0, 0), -1)
                        row_det_ctr_num[11] = row_det_ctr_num[11] + 1

        # compute y loc
        Z = 0.5
        # X = ((u-cx)/fx) * Z, Y = ((v-cy)/fy) * Z
        fx = 160.0
        fy = 160.0
        cx = 160.0
        cy = 120.0
        row_det_ctr_XY = np.zeros((12, 10, 2))

        for r in range(12):
            for n in range(10):
                if n<row_det_ctr_num[r]:
                    row_det_ctr_XY[r, n, 0] = ((row_det_ctr[r, n, 0] - cx) / fx) * Z
                    row_det_ctr_XY[r, n, 1] = ((row_det_ctr[r, n, 1] - cy) / fy) * Z

                    Y_plant_loc = int(row_det_ctr_XY[r, n, 1]*100) + 50 + int(self.plant_loc_offset)

                    if Y_plant_loc>=0 and Y_plant_loc<200:
                        self.plant_loc[r, Y_plant_loc] = 1
                    else:
                        rospy.logwarn('Y_plant_loc is out of range.')


        # self.axs[0].plot(self.plant_loc[0, :])
        # self.axs[1].plot(self.plant_loc[1, :])
        # plt.show()
        # print(self.plant_loc[0, 50: 200])

        # print(f"{self.plant_loc[0, 50: 200]}")

        nn_num = 3
        cmd = np.sum(self.plant_loc[:, 150 - nn_num: 150 + nn_num], axis=1)
        cmd[cmd>0] = 1
        print(cmd)

        weeder_cmd = np.zeros((13,))
        weeder_cmd[1:13] = cmd.copy()

        msg = Float32MultiArray()
        msg.data = weeder_cmd.tolist()
        self.weeder_control_pub.publish(msg)




                    # # (2) detect lanes
        # # extract lane_det_lines with the two step hough transform method
        # lines, seg_lines, rgb_lines = self.detect_lanes(plant_seg_gaus, cv_image)
        # if lines is None:
        #     rospy.logwarn('no lane_det_lines are detected, skipping this image and returning ...')
        #     return
        #
        # # publish line detection results
        # try:
        #     ros_seg_lines_image = self.bridge.cv2_to_imgmsg(seg_lines, "bgr8")
        #     ros_rgb_lines_image = self.bridge.cv2_to_imgmsg(rgb_lines, "bgr8")
        # except CvBridgeError as e:
        #     print(e)
        #     return
        # self.seg_line_pub.publish(ros_seg_lines_image)
        # # if self.cam_to_use == 0:
        # #     self.line_rgb_left_pub.publish(rgb_lines)
        # # elif self.cam_to_use==1:
        # #     self.line_rgb_right_pub.publish(rgb_lines)
        # self.rgb_line_pub.publish(ros_rgb_lines_image)
        #
        # t2 = rospy.get_time()
        # if self.verbose:
        #     rospy.loginfo('time consumption until line det = %f' % (t2 - t0))
        # if self.log_on:
        #     log_msg.data = str(rospy.get_time()) + ': finished line det.'
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #     log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t2 - t0)
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #
        # # (3) control weeder
        # # compute and send weeder cmd
        # weeder_cmd = self.control_weeder(lines)
        # if self.log_on:
        #     log_msg.data = str(rospy.get_time()) + ': current weeder_pos = ' + str(self.weeder_pos)
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #     log_msg.data = str(rospy.get_time()) + ': current weeder_speed = ' + str(self.weeder_speed)
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #     log_msg.data = str(rospy.get_time()) + ': lane center Y offset = ' + str(self.lane_y_offset)
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #     if weeder_cmd is not None:
        #         log_msg.data = str(rospy.get_time()) + ': new lane det weeder_cmd = ' +str(weeder_cmd)
        #         self.log_msg_pub.publish(log_msg)
        #         rospy.sleep(0.001)
        #     else:
        #         log_msg.data = str(rospy.get_time()) + ': no new lane det weeder_cmd sent. '
        #         self.log_msg_pub.publish(log_msg)
        #         rospy.sleep(0.001)
        #
        # t3 = rospy.get_time()
        # if self.verbose:
        #     rospy.loginfo('time consumption until weeder ctl = %f' % (t3 - t0))
        #     rospy.loginfo('++++++++++++')
        # if self.log_on:
        #     log_msg.data = str(rospy.get_time()) + ': finished sending weeder ctl.'
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #     log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t3 - t0)
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)
        #     log_msg.data = str(rospy.get_time()) + ': ++++++++++++'
        #     self.log_msg_pub.publish(log_msg)
        #     rospy.sleep(0.001)

        # return once everything successes
        return

    # segment plants using ExG
    def segment_plants_exg(self, np_image):
        # compute ExG index
        exg_index = 2 * np_image[:, :, 1] - (np_image[:, :, 2] + np_image[:, :, 0])  # ExG = 2G-(R+B)

        # Otsu thresholding
        ret, otsu_thr = cv2.threshold(exg_index, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # otsu_thr = self.param['lane_det']['exg_thr']
        image_seg_bn = (exg_index > otsu_thr).astype(float)

        # ma blob img
        img_seg_ma = image_seg_bn.copy()
        for n in range(3):
            img_seg_ma = cv2.GaussianBlur(img_seg_ma, (11, 11), 0)

        return image_seg_bn, np_image, img_seg_ma, None, None

    # segment plants using yolov5, return a binary image with plant area (1) and everything else (0)
    def segment_plants_yolo5(self, np_image):
        det_image = np_image.copy()
        image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

        # detection
        results = self.model(image_rgb)

        # create a binary image for segmentation result
        image_seg_bn = np.zeros((image_rgb.shape[0], image_rgb.shape[1]))

        # extract bbox coords, only select id=0, i.e. plants
        bboxes = results.xyxy[0].cpu().numpy()

        # convert bbox based object detection results to segmentation result, e.g. either draw rectangles or circles
        det_ctr = None
        det_bbox = None
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            if int(cls) == 0 and conf > 0.4 and y2 - y1 < 15 and x2 - x1 < 15:  # only select id=0, i.e. plants
                cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            if int(cls) == self.yolo5_plant_id:  # only select id=0, i.e. plants
                # image_seg_bn = cv2.rectangle(image_seg_bn, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1) # draw rect
                image_seg_bn = cv2.circle(image_seg_bn, (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
                                          int(np.min(np.array([x2 - x1, y2 - y1]) / 2.0)), 1, -1)  # draw circle
                cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                if det_ctr is None:
                    det_ctr = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                    det_bbox = np.array([[x1, y1, x2, y2]])
                else:
                    det_ctr = np.block([[det_ctr], [(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                    det_bbox = np.block([[det_bbox], [x1, y1, x2, y2]])

        # ma blob img
        # ma_det = scipy.ndimage.uniform_filter(image_seg_bn, size=3, axes=None)
        img_seg_ma = image_seg_bn.copy()
        for n in range(3):
            # img_seg_ma = scipy.ndimage.uniform_filter(img_seg_ma, size=10)  # don't use scipy.ndimage.uniform_filter, it will shift the location slightly
            img_seg_ma = cv2.GaussianBlur(img_seg_ma, (11, 11), 0)


        # return seg result, i.e. a binary image
        return image_seg_bn, det_image, img_seg_ma, det_ctr, det_bbox
    #
    # def detect_lanes(self, plant_seg_guas, cv_image):
    #     # wrap the raw image to let the middle lane in the middle of the image and vertically straight up, if needed.
    #     if self.wrap_img_on:
    #         wrap_param = self.wrap_param
    #
    #         # fill two sides of the image will black image with half of its width
    #         u_half_size = int(self.img_size[1] / 2.0)
    #         cv_image = np.concatenate((cv_image[:, 0:u_half_size, :] * 0, cv_image, cv_image[:, 0:u_half_size, :] * 0), axis=1)
    #         plant_seg_guas = np.concatenate((plant_seg_guas[:, 0:u_half_size] * 0, plant_seg_guas, plant_seg_guas[:, 0:u_half_size] * 0), axis=1)
    #         # crop image, 1st vertical then 2nd horizontal
    #         cv_image = cv_image[wrap_param[5]:(wrap_param[4] + 1), :, :]  # vertical crop
    #         cv_image = cv_image[:, int(u_half_size + wrap_param[0] - 0.5 * (wrap_param[1] - wrap_param[0])):
    #                                int(u_half_size + wrap_param[1] + 0.5 * (wrap_param[1] - wrap_param[0])), :]
    #         plant_seg_guas = plant_seg_guas[wrap_param[5]:(wrap_param[4] + 1), :]  # horizontal crop
    #         plant_seg_guas = plant_seg_guas[:, int(u_half_size + wrap_param[0] - 0.5 * (wrap_param[1] - wrap_param[0])):
    #                                            int(u_half_size + wrap_param[1] + 0.5 * (wrap_param[1] - wrap_param[0]))]
    #         # transform
    #         cv_image = cv2.warpPerspective(cv_image, self.wrap_H, (cv_image.shape[1], cv_image.shape[0]))
    #         plant_seg_guas = cv2.warpPerspective(plant_seg_guas, self.wrap_H, (plant_seg_guas.shape[1], plant_seg_guas.shape[0]))
    #
    #         # resize
    #         cv_image_bf_resize = cv_image.copy()  # save img before resize
    #         plant_seg_guas_bf_resize = plant_seg_guas.copy()  # save img before resize
    #         cv_image = cv2.resize(cv_image, (int(self.wrap_img_size_u), int(cv_image.shape[0] * (self.wrap_img_size_u / cv_image.shape[1]))))
    #         plant_seg_guas = cv2.resize(plant_seg_guas, (int(self.wrap_img_size_u), int(plant_seg_guas.shape[0] * (self.wrap_img_size_u / plant_seg_guas.shape[1]))))
    #     else:
    #         cv_image_bf_resize = cv_image.copy()  # save img before resize
    #
    #     img_size = plant_seg_guas.shape  # image size after wrapping
    #
    #     # track previous line det results
    #     if self.lane_det_track_img is None:
    #         self.lane_det_track_img = np.zeros((img_size[0], img_size[1]))
    #     track_on = self.param['lane_det']['track_on']
    #     track_img_pre_weight = self.param['lane_det']['track_img_pre_weight']
    #     if track_on:
    #         if np.max(self.lane_det_track_img)>0.01:
    #             plant_seg_guas = plant_seg_guas + self.lane_det_track_img * track_img_pre_weight  # TODO
    #             plant_seg_guas[plant_seg_guas>1] = 1
    #
    #     # prepare vars for searching best two parallel lines
    #     x_range = np.array([x for x in range(int(img_size[1]/2.0))])  # u ranges in wrapped image
    #     y_range = np.array([x for x in range(img_size[0])])  # v ranges in wrapped image
    #     # left line's theta angle from wrap_param, i.e. to find ref's left line theta
    #     ll_theta_from_wrap_param = -np.rad2deg(np.arctan2(((self.wrap_param[1]-self.wrap_param[0])/2.0) - ((self.wrap_param[3]-self.wrap_param[2])/2.0),
    #                                                       (self.wrap_param[4] - self.wrap_param[5])))
    #     theta_range = np.array([theta for theta in range(-self.param['lane_det']['theta_range'], self.param['lane_det']['theta_range'] + 1)]) \
    #                   + ll_theta_from_wrap_param  # theta range to search from
    #
    #     # ref left and right lines' intersection point's v coord dist from the bottom of the image
    #     inter_h = ((self.wrap_param[1]-self.wrap_param[0]) / ((self.wrap_param[1]-self.wrap_param[0]) - (self.wrap_param[3]-self.wrap_param[2]))) * self.wrap_img_size[0]
    #     shift_range = np.array([x for x in range(-self.param['lane_det']['shift_range'], self.param['lane_det']['theta_range'] + 1)])
    #
    #     # construct 3D hist for search best left and right lines,
    #     # his_3d: [search left line's bot u, left line's theta, right line's shift from left line's bot u + ref lane widht]
    #     his_3d = np.zeros((x_range.shape[0], theta_range.shape[0], shift_range.shape[0]))
    #
    #     # prepare vars for C++ based hist computation operation
    #     cvar_img_size_u = int(self.wrap_img_size[1])
    #     cvar_img_size_v = int(self.wrap_img_size[0])
    #     cvar_x_range_num = int(x_range.shape[0])
    #     cvar_y_range_num = int(y_range.shape[0])
    #     cvar_theta_range = theta_range.astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    #     cvar_theta_range_num = int(theta_range.shape[0])
    #     cvar_shift_range = shift_range.astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    #     cvar_shift_range_num = int(shift_range.shape[0])
    #     cvar_inter_h = ctypes.c_double(inter_h)
    #     cvar_plant_seg_guas = plant_seg_guas.reshape((-1,)).astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # row first
    #     np_cvar_his = his_3d.reshape((-1,)).astype(np.double)
    #
    #     # compute hist 3D: np_cvar_his, using C++
    #     self.cfunc_handle.GrayScaleHoughLine(cvar_img_size_u, cvar_img_size_v, cvar_x_range_num, cvar_y_range_num, cvar_theta_range, cvar_theta_range_num,
    #                                          cvar_shift_range, cvar_shift_range_num, cvar_inter_h,
    #                                          cvar_plant_seg_guas, np_cvar_his.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    #
    #     his_3d = np_cvar_his.reshape((x_range.shape[0], theta_range.shape[0], shift_range.shape[0])).copy()  # recover its 3D shape
    #
    #     # ref lines' bot (down) pixel's  and top (up) pixel's u
    #     ref_lines_u_d = np.array([self.wrap_img_size_u*0.25, self.wrap_img_size_u*0.75])
    #     ref_lines_u_u = np.array([self.wrap_img_size_u*0.5 - ((self.wrap_param[3]-self.wrap_param[2])/2.0)*(self.wrap_img_size_u/((self.wrap_param[1]-self.wrap_param[0])*2.0)),
    #                               self.wrap_img_size_u*0.5 + ((self.wrap_param[3]-self.wrap_param[2])/2.0)*(self.wrap_img_size_u/((self.wrap_param[1]-self.wrap_param[0])*2.0))])
    #     ref_lines_theta = np.array([np.rad2deg(np.arctan2(ref_lines_u_d[0] - ref_lines_u_u[0], self.wrap_img_size[0])),
    #                                 np.rad2deg(np.arctan2(ref_lines_u_d[1] - ref_lines_u_u[1], self.wrap_img_size[0]))])
    #     ref_lines_idx = np.array([np.argmin(np.abs(x_range - ref_lines_u_d[0])), np.argmin(np.abs(theta_range - ref_lines_theta[0])), (shift_range.shape[0]-1)/2.0])
    #     # ref lines' heuristic score
    #     ref_lines_heur = his_3d[int(ref_lines_idx[0]), int(ref_lines_idx[1]), int(ref_lines_idx[2])]
    #
    #     # from previous line det results, only search left line's bot u from the local region of the previous det result
    #     his_3d_u_track = his_3d.copy() * 0
    #     track_u_width = self.param['lane_det']['track_u_width']
    #     if self.lane_det_track_u is not None:
    #         his_3d_u_track[int(max(0, self.lane_det_track_u[0]-track_u_width)) : int(min(self.lane_det_track_u[0]+track_u_width, img_size[1])), :, :] = \
    #             his_3d[int(max(0, self.lane_det_track_u[0]-track_u_width)) : int(min(self.lane_det_track_u[0]+track_u_width, img_size[1])), :, :].copy()
    #     else:
    #         his_3d_u_track = his_3d.copy()
    #     # get the best two lines
    #     max_idx = np.unravel_index(np.argmax(his_3d_u_track), his_3d_u_track.shape)
    #     track_lines_heur = his_3d[max_idx[0], max_idx[1], max_idx[2]]
    #
    #     if track_lines_heur>ref_lines_heur:
    #         # collect left and right lines' params
    #         ll_u = x_range[max_idx[0]]  # left line's bot u
    #         ll_theta = theta_range[max_idx[1]]  # left line's theta
    #         rl_u = ll_u + self.wrap_img_size[1]/2.0 + shift_range[max_idx[2]]  # right line's bot u
    #         inter_u = ll_u + inter_h * np.tan(-np.deg2rad(ll_theta))  # left and right lines intersection point's u coord
    #         rl_theta = np.rad2deg(np.arctan2(rl_u - inter_u, inter_h))  # right line's theta
    #
    #         lane_u_d = np.array([ll_u, rl_u])  # two lines' bottom points' u coord
    #         lane_theta = np.array([ll_theta, rl_theta])  # two lines' theta
    #         # two lines' upper points' u coord
    #         lane_u_u = np.array([-np.tan(np.deg2rad(lane_theta[0]))*img_size[0] + lane_u_d[0], -np.tan(np.deg2rad(lane_theta[1]))*img_size[0] + lane_u_d[1]])
    #     else:
    #         lane_u_d = ref_lines_u_d.copy()  # two lines' bottom points' u coord
    #         lane_theta = ref_lines_theta.copy()  # two lines' theta
    #         lane_u_u = ref_lines_u_u.copy()  # two lines' upper points' u coord
    #
    #     if self.weeder_status==0:  # if weeder is up, don't detect lines
    #         lane_u_d = ref_lines_u_d.copy()  # two lines' bottom points' u coord
    #         lane_theta = ref_lines_theta.copy()  # two lines' theta
    #         lane_u_u = ref_lines_u_u.copy()  # two lines' upper points' u coord
    #
    #
    #     # plant seg + lines image
    #     plant_seg_guas_lines = np.stack(((plant_seg_guas * 255.0).astype(np.uint8),) * 3, axis=-1)
    #     # plant_seg_guas_lines = np.stack((plant_seg_guas,) * 3, axis=-1)
    #     cv2.line(plant_seg_guas_lines, (int(lane_u_d[0]), img_size[0]-1), (int(lane_u_u[0]), 0), (0, 255, 0), 1, cv2.LINE_AA)
    #     cv2.line(plant_seg_guas_lines, (int(lane_u_d[1]), img_size[0]-1), (int(lane_u_u[1]), 0), (0, 255, 0), 1, cv2.LINE_AA)
    #     cv2.line(plant_seg_guas_lines, (int(ref_lines_u_d[0]), img_size[0]-1), (int(ref_lines_u_u[0]), 0), (255, 0, 0), 1, cv2.LINE_AA)
    #     cv2.line(plant_seg_guas_lines, (int(ref_lines_u_d[1]), img_size[0]-1), (int(ref_lines_u_u[1]), 0), (255, 0, 0), 1, cv2.LINE_AA)
    #
    #     # raw rgb image + lines image
    #     # cv_image_lines = cv_image.copy()
    #     # cv2.line(cv_image_lines, (int(lane_u_d[0]), img_size[0] - 1), (int(lane_u_u[0]), 0), (0, 255, 0), 1, cv2.LINE_AA)
    #     # cv2.line(cv_image_lines, (int(lane_u_d[1]), img_size[0] - 1), (int(lane_u_u[1]), 0), (0, 255, 0), 1, cv2.LINE_AA)
    #     # cv2.line(cv_image_lines, (int(ref_lines_u_d[0]), img_size[0]-1), (int(ref_lines_u_u[0]), 0), (255, 0, 0), 1, cv2.LINE_AA)
    #     # cv2.line(cv_image_lines, (int(ref_lines_u_d[1]), img_size[0]-1), (int(ref_lines_u_u[1]), 0), (255, 0, 0), 1, cv2.LINE_AA)
    #     cv_image_lines = cv_image_bf_resize.copy()  # show rgb image with better resolution
    #     cv2.line(cv_image_lines, (int(lane_u_d[0]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
    #              (int(lane_u_u[0]*(1.0/self.resize_proportion)), 0), (0, 255, 0), 1, cv2.LINE_AA)
    #     cv2.line(cv_image_lines, (int(lane_u_d[1]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
    #              (int(lane_u_u[1]*(1.0/self.resize_proportion)), 0), (0, 255, 0), 1, cv2.LINE_AA)
    #     cv2.line(cv_image_lines, (int(ref_lines_u_d[0]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
    #              (int(ref_lines_u_u[0]*(1.0/self.resize_proportion)), 0), (255, 0, 0), 1, cv2.LINE_AA)
    #     cv2.line(cv_image_lines, (int(ref_lines_u_d[1]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
    #              (int(ref_lines_u_u[1]*(1.0/self.resize_proportion)), 0), (255, 0, 0), 1, cv2.LINE_AA)
    #
    #     lines = np.block([[lane_u_d], [lane_u_u]]) # down pixel's  and up pixel's u for a line
    #
    #     # re-generate tracking image, i.e. add two white lines on the black background image
    #     self.lane_det_track_img = self.lane_det_track_img * 0
    #     cv2.line(self.lane_det_track_img, (int(lane_u_d[0]), img_size[0] - 1), (int(lane_u_u[0]), 0), 1, self.param['lane_det']['track_line_width'], cv2.LINE_AA)
    #     cv2.line(self.lane_det_track_img, (int(lane_u_d[1]), img_size[0] - 1), (int(lane_u_u[1]), 0), 1, self.param['lane_det']['track_line_width'], cv2.LINE_AA)
    #     # blur the two white lines, don't use scipy.ndimage.uniform_filter or cv2.blur, they will shift the position of the line
    #     for n in range(5):
    #         self.lane_det_track_img = cv2.GaussianBlur(self.lane_det_track_img, (self.param['lane_det']['track_line_filter_width'], self.param['lane_det']['track_line_filter_width']), 0)
    #
    #     # save relevant vars
    #     self.lane_det_lines = lines.copy()
    #     self.lane_det_track_u = self.lane_det_lines[0, :]
    #
    #     # return results
    #     return lines, plant_seg_guas_lines, cv_image_lines  # line: 2x1, down pixel's  and up pixel's u for a line
    #
    # # weeder control
    # def control_weeder(self, lines):
    #     wrap_param = self.wrap_param
    #
    #     lines_u_d = lines[0, :]
    #     lane_u_mid = (lines_u_d[0] + lines_u_d[1])/2.0
    #
    #     weeder_pos_pix = (wrap_param[6]-wrap_param[0]+int((wrap_param[1]-wrap_param[0])*0.5)) * (self.wrap_img_size_u/((wrap_param[1]-wrap_param[0])*2))
    #     print('weeder_pos_pix = %f' % weeder_pos_pix)
    #
    #     lane_u_offset = lane_u_mid - weeder_pos_pix
    #     print('lane_u_offset = %f' % lane_u_offset)
    #     # lane_u_offset = lane_u_mid-(self.wrap_img_size_u / 2.0)
    #
    #     # y coord of the machine is pointing to left
    #     if self.cam_to_use == 0:
    #         lane_y_offset = -lane_u_offset*(self.param['weeder']['farm_lane_dist_left_cam']/(lines_u_d[1]-lines_u_d[0]))
    #     elif self.cam_to_use == 1:
    #         lane_y_offset = -lane_u_offset*(self.param['weeder']['farm_lane_dist_right_cam']/(lines_u_d[1]-lines_u_d[0]))
    #     else:
    #         rospy.logwarn('lane_det: unknown camera selection.')
    #         return
    #
    #     # apply max shift
    #     if lane_y_offset > self.param['weeder']['weeder_cmd_max_shift']:
    #         lane_y_offset = self.param['weeder']['weeder_cmd_max_shift']
    #     elif lane_y_offset < (-self.param['weeder']['weeder_cmd_max_shift']):
    #         lane_y_offset = -self.param['weeder']['weeder_cmd_max_shift']
    #
    #     self.lane_y_offset = lane_y_offset
    #     weeder_cmd = lane_y_offset + self.weeder_pos
    #
    #     # after a fixed distance, send weeder control cmd
    #     cur_time = rospy.get_time()
    #     if (cur_time - self.ctl_time_pre) * self.weeder_speed > self.param['weeder']['cmd_dist']:
    #
    #         # fill the weeder_cmd_buff
    #         step_num = int(np.floor((cur_time - self.ctl_time_pre) * self.weeder_speed / self.param['weeder']['cmd_dist']))
    #         for step in range(step_num):
    #             self.weeder_cmd_buff[0:self.weeder_cmd_buff.shape[0] - 1] = self.weeder_cmd_buff[1:self.weeder_cmd_buff.shape[0]].copy()
    #             self.weeder_cmd_buff[self.weeder_cmd_buff.shape[0] - 1] = weeder_cmd
    #
    #         # use delayed weeder cmd or the newest weeder cmd
    #         if self.weeder_cmd_delay:
    #             self.weeder_cmd = self.weeder_cmd_buff[0]
    #         else:
    #             # self.weeder_cmd = self.weeder_cmd_buff[-1]  # extract the most recent position shift
    #             self.weeder_cmd = weeder_cmd
    #
    #         self.ctl_time_pre = cur_time
    #
    #         # send the control cmd, i.e. the absolute position shift of the weeder
    #         msg = Float32MultiArray()
    #         msg.data = [self.weeder_cmd]
    #         self.weeder_control_pub.publish(msg)
    #         if self.verbose:
    #             rospy.loginfo('sent weeder_cmd = %f' % (self.weeder_cmd))
    #         return self.weeder_cmd
    #     else:
    #         return None

    # resize image to the target size, but keep the horizontal and vertical scale. crop image if needed
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

        # resize to target size
        img_resize = cv2.resize(img_crop, (self.img_size[1], self.img_size[0]))

        # return the resized image
        return img_resize


# the main function entrance
def main(args):
    rospy.init_node('intra_row_node', anonymous=True)
    intra_row_obj = intra_row()

    # the main loop for line detection
    while not rospy.is_shutdown():
        intra_row_obj.img_cb()


# the main entrance
if __name__ == '__main__':
    main(sys.argv)

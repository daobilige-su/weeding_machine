#! /usr/bin/env python
import numpy as np
import rospy
import sys

import scipy.ndimage
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

from scipy.stats import multivariate_normal
from scipy import signal

import torch

import rospkg
import yaml

import time
import ctypes

class image_processor:
    def __init__(self):
        # locate ros pkg
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'

        # load key params
        params_filename = rospy.get_param('param_file')  # self.pkg_path + 'cfg/' + 'param.yaml'
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)
        self.verbose = self.param['lane_det']['verbose']
        self.img_size = np.array([self.param['cam']['height'], self.param['cam']['width']])
        self.bird_pixel_size = self.param['lane_det']['bird_pixel_size']
        self.max_lane_width = self.param['lane_det']['max_lane_width']
        self.bird_roi_m = self.param['lane_det']['bird_roi']  # in weeder base_link coord, [y_max, y_min, x_max, x_min]

        # yolo5 detection
        self.yolo5_plant_id = self.param['lane_det']['yolo_plant_id']
        self.yolo5_model_path = self.pkg_path + 'weights/' + self.param['lane_det']['yolo_model']
        self.model = torch.hub.load(self.pkg_path + 'yolov5', 'custom', path=self.yolo5_model_path, source='local')

        # cam vars, pubs and subs
        self.left_img = None
        self.right_img = None
        self.bridge = CvBridge()
        self.cam_to_use = self.param['cam']['cam_to_use']
        self.cam_sel_time_interval = self.param['cam']['cam_sel_time_interval']
        self.choose_cam_pre_time = rospy.get_time()
        if self.cam_to_use == -1:
            if not self.select_cam():
                rospy.logwarn('camera auto selection failed.')
                return
        self.left_img_sub = rospy.Subscriber(self.param['cam']['left']['topic'], Image, self.left_img_cb, queue_size=1)
        self.right_img_sub = rospy.Subscriber(self.param['cam']['right']['topic'], Image, self.right_img_cb, queue_size=1)
        self.proc_img_pub = rospy.Publisher('/proc_img/image_raw', Image, queue_size=2)
        self.seg_img_pub = rospy.Publisher('/seg_img/image_raw', Image, queue_size=2)
        self.line_img_pub = rospy.Publisher('/line_img/image_raw', Image, queue_size=2)
        self.det_img_left_pub = rospy.Publisher('/det_img/image_raw_left', Image, queue_size=2)
        self.det_img_right_pub = rospy.Publisher('/det_img/image_raw_right', Image, queue_size=2)
        self.line_rgb_left_pub = rospy.Publisher('/line_rgb/iamge_raw_left', Image, queue_size=2)
        self.line_rgb_right_pub = rospy.Publisher('/line_rgb/iamge_raw_right', Image, queue_size=2)

        # weeder pub, sub and vars
        self.weeder_speed = self.param['weeder']['def_speed']
        self.weeder_pos = self.param['weeder']['def_pos']  # TODO
        self.weeder_control_pub = rospy.Publisher('/weeder_cmd', Float32MultiArray, queue_size=2)
        self.weeder_speed_pos_sub = rospy.Subscriber('/weeder_speed_pos', Float32MultiArray, self.weeder_speed_pos_cb)  # TODO
        self.mid_line_y = None
        self.control_bias = self.param['weeder']['ctrl_bias']
        self.weeder_y = 0
        self.mid_y_buff = np.zeros((self.param['weeder']['cmd_buffer_size'],))
        self.weeder_pos_buff = np.zeros((self.param['weeder']['cmd_buffer_size'],))  # TODO
        self.ctl_time_pre = -1
        self.ctl_cmd = 0
        self.weeder_cmd_delay = self.param['weeder']['weeder_cmd_delay']  # TODO

        # get camera intrinsic
        self.left_cam_K = np.array(self.param['cam']['left']['K']).reshape((3, 3))
        self.right_cam_K = np.array(self.param['cam']['right']['K']).reshape((3, 3))
        # get camera extrinsic
        self.left_cam_T = transform_trans_ypr_to_matrix(np.array(self.param['cam']['left']['T']).reshape((6, 1)))
        self.right_cam_T = transform_trans_ypr_to_matrix(np.array(self.param['cam']['right']['T']).reshape((6, 1)))
        # camera homography
        self.left_bird_img_size = None
        self.right_bird_img_size = None
        self.left_cam_H = None
        self.right_cam_H = None
        # computes self.left_bird_img_size, self.right_bird_img_size, self.left_cam_H, self.right_cam_H
        # Note that left_bird_img_size and right_bird_img_size is in the order of (u,v), not (v,u) in image!!
        self.left_bird_img_size, self.left_cam_H = self.compute_homography(self.left_cam_K, self.left_cam_T)
        self.right_bird_img_size, self.right_cam_H = self.compute_homography(self.right_cam_K, self.right_cam_T)
        # bird eye view roi in meter to that in percentage
        self.left_bird_roi_perc = [
            (-self.bird_roi_m[0] / self.bird_pixel_size + self.left_bird_img_size[0] / 2.0) / self.left_bird_img_size[
                0],
            (-self.bird_roi_m[1] / self.bird_pixel_size + self.left_bird_img_size[0] / 2.0) / self.left_bird_img_size[
                0],
            1 - ((self.bird_roi_m[2] / self.bird_pixel_size) / self.left_bird_img_size[1]),
            1 - ((self.bird_roi_m[3] / self.bird_pixel_size) / self.left_bird_img_size[1])]
        self.right_bird_roi_perc = [
            (-self.bird_roi_m[0] / self.bird_pixel_size + self.right_bird_img_size[0] / 2.0) / self.right_bird_img_size[
                0],
            (-self.bird_roi_m[1] / self.bird_pixel_size + self.right_bird_img_size[0] / 2.0) / self.right_bird_img_size[
                0],
            1 - ((self.bird_roi_m[2] / self.bird_pixel_size) / self.right_bird_img_size[1]),
            1 - ((self.bird_roi_m[3] / self.bird_pixel_size) / self.right_bird_img_size[1])]
        # cpp function call
        self.cfunc_handle = ctypes.CDLL(self.pkg_path + "libGrayScaleHoughLine.so")
        self.lane_det_track_img = None
        self.lane_det_track_u = None
        self.lane_det_lines = None

        self.wrap_param = np.array([180, 319, 200, 0, 239, 250])
        self.wrap_img_size = np.array([int(self.wrap_param[4]-self.wrap_param[3]+1), int(2*(self.wrap_param[1]-self.wrap_param[0]))])
        h_shift = (self.wrap_param[0] + self.wrap_param[1]) / 2.0 - self.wrap_param[2]
        if h_shift > 0:
            inputpts = np.float32([[0, self.img_size[0]], [self.img_size[1], self.img_size[0]], [self.img_size[1] - int(h_shift), 0],[0, 0]])
            outputpts = np.float32([[0, self.img_size[0]], [self.img_size[1], self.img_size[0]], [self.img_size[1], 0], [int(h_shift), 0]])
        else:
            inputpts = np.float32([[0, self.img_size[0]], [self.img_size[1], self.img_size[0]], [self.img_size[1], 0], [int(h_shift), 0]])
            outputpts = np.float32([[0, self.img_size[0]], [self.img_size[1], self.img_size[0]], [self.img_size[1] - int(h_shift), 0],[0, 0]])
        self.wrap_H = cv2.getPerspectiveTransform(inputpts, outputpts)

        return

    # TODO, sub for weeder weeder speed and pos, simply store it in member var
    def weeder_speed_pos_cb(self, msg):
        self.weeder_speed = msg.data[0]
        self.weeder_pos = msg.data[1]

    # sub for left cam img, store it in member var and call line detection if left cam is selected.
    def left_img_cb(self, msg):
        # store var
        try:
            self.left_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(msg, "bgr8"))
        except CvBridgeError as e:
            print(e)
            return

    # sub for right cam img, store it in member var and call line detection if right cam is selected.
    def right_img_cb(self, msg):
        # store var
        try:
            self.right_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(msg, "bgr8"))
        except CvBridgeError as e:
            print(e)
            return

    # select best camera based on the number plants detected and the total plant area
    def select_cam(self):
        rospy.logwarn('auto selection of camera started.')

        # obtain camera images
        rospy.logwarn('waiting 100s for left and right camera images ...')
        try:
            left_img_msg = rospy.wait_for_message(self.param['cam']['left']['topic'], Image, 100)
        except ROSException as e:
            rospy.logwarn('lane_det: left image not received after 100s.')
            rospy.logwarn(e)
            return False
        try:
            right_img_msg = rospy.wait_for_message(self.param['cam']['right']['topic'], Image, 100)
        except ROSException as e:
            rospy.logwarn(e)
            rospy.logwarn('lane_det: right image not received after 100s.')
            return False
        rospy.logwarn('both left and right camera images received. ')

        left_cv_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8"))
        right_cv_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8"))

        # get plant segmentation images
        left_image_seg_bn, image1, left_image_seg_gaus = self.segment_plants_yolo5(left_cv_img)
        right_image_seg_bn, image2, right_image_seg_gaus = self.segment_plants_yolo5(right_cv_img)
        # compute plant area
        left_plant_area = np.sum(left_image_seg_bn)
        right_plant_area = np.sum(right_image_seg_bn)

        # determine camera to use
        if (left_plant_area>=right_plant_area) and \
                (left_plant_area>=self.param['cam']['cam_sel_min_perc_plant']*self.img_size[0]*self.img_size[1]):
            self.cam_to_use = 0
            rospy.logwarn('left camera is selected.')
        elif (right_plant_area>left_plant_area) and \
                (right_plant_area>self.param['cam']['cam_sel_min_perc_plant']*self.img_size[0]*self.img_size[1]):
            self.cam_to_use = 1
            rospy.logwarn('right camera is selected.')
        else:
            rospy.logwarn('lane_det: not enough plant area detected when selecting camera.')
            return False

        return True

    # the main loop: detect line and send back the result
    def img_cb(self):
        t0 = time.time()
        # print('t1 = %f' % (t1-t0))

        #choose camera in each 0.5s
        choose_cam_cur_time = rospy.get_time()
        if choose_cam_cur_time - self.choose_cam_pre_time >= self.cam_sel_time_interval:
            self.select_cam()
            self.choose_cam_pre_time = rospy.get_time()
        # select left or right camera image for lane detection
        if self.cam_to_use == 0:
            cv_image = self.left_img
        elif self.cam_to_use == 1:
            cv_image = self.right_img
        else:
            rospy.logwarn('lane_det: unknown camera selection.')
            return

        if cv_image is None:
            return
        else:
            # publish image to be processed
            try:
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            except CvBridgeError as e:
                print(e)
                return
            self.proc_img_pub.publish(ros_image)

        t01 = time.time()
        print('t0 = %f' % (t01-t0))

        # (1) segment plants in the image
        # segment plants with yolov5
        plant_seg_bn, det_image, plant_seg_gaus = self.segment_plants_yolo5(cv_image)

        t11 = time.time()
        print('t11 = %f' % (t11 - t0))

        # publish segmentation results
        plant_seg_pub = np.asarray((plant_seg_bn * 255).astype(np.uint8))
        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_pub, "mono8")
            det_image = self.bridge.cv2_to_imgmsg(det_image,"bgr8")
        except CvBridgeError as e:
            print(e)
            return
        self.seg_img_pub.publish(ros_image)
        if self.cam_to_use==0:
            self.det_img_left_pub.publish(det_image)
        elif self.cam_to_use==1:
            self.det_img_right_pub.publish(det_image)

        t1 = time.time()
        print('t1 = %f' % (t1 - t0))

        # (2) detect lanes
        # extract lane_det_lines with the two step hough transform method
        # lane_det_lines, plant_seg_lines, rgb_lines = self.detect_lanes(plant_seg_bn, cv_image)
        lines, plant_seg_lines, rgb_lines = self.detect_lanes_gaus(plant_seg_gaus, cv_image)
        if lines is None:
            rospy.logwarn('no lane_det_lines are detected, skipping this image and returning ...')
            return

        # publish line detection results
        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_lines, "bgr8")
            rgb_lines = self.bridge.cv2_to_imgmsg(rgb_lines, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.line_img_pub.publish(ros_image)
        if self.cam_to_use == 0:
            self.line_rgb_left_pub.publish(rgb_lines)
        elif self.cam_to_use==1:
            self.line_rgb_right_pub.publish(rgb_lines)

        t2 = time.time()
        print('t2 = %f' % (t2 - t0))

        # (3) control weeder
        # compute and send weeder cmd
        # self.control_weeder(lane_det_lines)
        self.control_weeder_gaus(lines)

        t3 = time.time()
        print('t3 = %f' % (t3 - t0))
        print('++++++++++++')

        # return once everything successes
        return

    # segment plants using yolov5, return a binary image with plant area (1) and everything else (0)
    # def segment_plants_yolo5(self, np_image):
    #     det_image = np_image.copy()
    #     image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    #
    #     # detection
    #     t0 = time.time()
    #     results = self.model(image_rgb)
    #     t1 = time.time()
    #     print('yolo t: %f' % (t1-t0))
    #
    #     # create a binary image for segmentation result
    #     image_seg_bn = np.zeros((image_rgb.shape[0], image_rgb.shape[1]))
    #
    #     # extract bbox coords, only select id=0, i.e. plants
    #     bboxes = results.xyxy[0].cpu().numpy()
    #
    #     # gaussian blob img
    #     bboxes_num = bboxes.shape[0]
    #     gaus_det = np.zeros((self.img_size[0], self.img_size[1], bboxes_num))
    #     x, y = np.mgrid[0:self.img_size[0], 0:self.img_size[1]]
    #     gaus_layer_idx = np.dstack((x, y))
    #
    #     # convert bbox based object detection results to segmentation result, e.g. either draw rectangles or circles
    #     bboxes_idx = 0
    #     for bbox in bboxes:
    #         x1, y1, x2, y2, conf, cls = bbox
    #         if int(cls) == 0 and conf > 0.4 and y2 - y1 < 15 and x2 - x1 < 15:  # only select id=0, i.e. plants
    #             cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    #         if int(cls) == self.yolo5_plant_id:  # only select id=0, i.e. plants
    #             # image_seg_bn = cv2.rectangle(image_seg_bn, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1) # draw rect
    #             image_seg_bn = cv2.circle(image_seg_bn, (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
    #                                       int(np.min(np.array([x2 - x1, y2 - y1]) / 2.0)), 1, -1)  # draw circle
    #             cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    #
    #             gaus_pdf = multivariate_normal([(y1 + y2) / 2.0, (x1 + x2) / 2.0], [[(0.6*(y2-y1))**2, 0], [0, (0.6*(x2-x1))**2]])
    #             gaus_layer = gaus_pdf.pdf(gaus_layer_idx)
    #             gaus_layer = gaus_layer*(1.0/np.max(gaus_layer))
    #
    #             gaus_det[:, :, bboxes_idx] = gaus_layer.copy()
    #             bboxes_idx = bboxes_idx+1
    #
    #     # gaussian blob image
    #     img_seg_gaus = np.max(gaus_det, axis=2)
    #     # img_seg_gaus = np.sum(gaus_det, axis=2)
    #     img_seg_gaus = img_seg_gaus*(255.0/np.max(img_seg_gaus))
    #     img_seg_gaus = img_seg_gaus.astype(np.uint8)
    #
    #     # return seg result, i.e. a binary image
    #     return image_seg_bn, det_image, img_seg_gaus
    def segment_plants_yolo5(self, np_image):
        det_image = np_image.copy()
        image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

        # detection
        t0 = time.time()
        results = self.model(image_rgb)
        t1 = time.time()
        print('yolo t: %f' % (t1-t0))

        # create a binary image for segmentation result
        image_seg_bn = np.zeros((image_rgb.shape[0], image_rgb.shape[1]))

        # extract bbox coords, only select id=0, i.e. plants
        bboxes = results.xyxy[0].cpu().numpy()

        # convert bbox based object detection results to segmentation result, e.g. either draw rectangles or circles
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            if int(cls) == 0 and conf > 0.4 and y2 - y1 < 15 and x2 - x1 < 15:  # only select id=0, i.e. plants
                cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            if int(cls) == self.yolo5_plant_id:  # only select id=0, i.e. plants
                # image_seg_bn = cv2.rectangle(image_seg_bn, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1) # draw rect
                image_seg_bn = cv2.circle(image_seg_bn, (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
                                          int(np.min(np.array([x2 - x1, y2 - y1]) / 2.0)), 1, -1)  # draw circle
                cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # mpl.use('TkAgg')
        # plt.imshow(image_seg_bn, cmap='gray')
        # plt.show()

        # ma blob img
        # ma_det = scipy.ndimage.uniform_filter(image_seg_bn, size=3, axes=None)
        ma_det = image_seg_bn.copy()
        for n in range(3):
            ma_det = scipy.ndimage.uniform_filter(ma_det, size=10)

        # img_seg_ma = ma_det * (255.0 / np.max(ma_det))
        # img_seg_ma = img_seg_ma.astype(np.uint8)
        img_seg_ma = ma_det.copy()

        # return seg result, i.e. a binary image
        return image_seg_bn, det_image, img_seg_ma

    def detect_lanes_gaus(self, plant_seg_guas, cv_image):
        # lane_det_lines = cv2.HoughLines(plant_seg_guas, 50, np.deg2rad(1), self.param['lane_det']['hough_hist_thr'], None, 0, 0,
        #                        np.deg2rad(self.param['lane_det']['hough_theta_range'][0]),
        #                        np.deg2rad(self.param['lane_det']['hough_theta_range'][1]))
        #
        # plant_seg_guas_lines = np.stack((plant_seg_guas,) * 3, axis=-1)
        #
        # lines2d = lane_det_lines.reshape((lane_det_lines.shape[0], lane_det_lines.shape[2]))
        #
        # for i in range(lines2d.shape[0]):
        #     rho = lines2d[i, 0]
        #     theta = lines2d[i, 1]
        #     a = math.cos(theta)
        #     b = math.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #     pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #     cv2.line(plant_seg_guas_lines, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        #     pt_edge_x = (rho - plant_seg_guas.shape[0] * b) / a
        #     # lines_x_theta[0, i] = pt_edge_x
        #     cv2.circle(plant_seg_guas_lines, (int(plant_seg_guas_lines.shape[1] / 2.0), plant_seg_guas_lines.shape[0]), 5,
        #                (255, 0, 0), 1)
        #     cv2.circle(plant_seg_guas_lines, (int(pt_edge_x), plant_seg_guas_lines.shape[0]), 5, (0, 0, 255), 1)
        #
        #     mpl.use('TkAgg')
        #     plt.imshow(plant_seg_guas_lines)
        #     plt.show()
        # t0 = time.time()

        # plant_seg_guas_lines = np.stack(((plant_seg_guas * (255.0 / np.max(plant_seg_guas))).astype(np.uint8),) * 3, axis=-1)

        wrap_param = self.wrap_param
        wrap_H = self.wrap_H

        wrap_img_on = 1
        if wrap_img_on:
            ori_img_size = plant_seg_guas.shape
            u_half_size = int(ori_img_size[1]/2.0)

            # cv_image
            cv_image = cv_image[wrap_param[3]:(wrap_param[4]+1), :, :]
            cv_image = np.concatenate((cv_image[:, 0:u_half_size, :]*0, cv_image, cv_image[:, 0:u_half_size, :]*0), axis=1)

            cv_image = cv_image[:, int(u_half_size+wrap_param[0] - 0.5*(wrap_param[1]-wrap_param[0])):int(u_half_size+wrap_param[1] + 0.5*(wrap_param[1]-wrap_param[0])), :]

            cv_image = cv2.warpPerspective(cv_image, wrap_H, (cv_image.shape[1], cv_image.shape[0]))

            # plant_seg_guas
            plant_seg_guas = plant_seg_guas[wrap_param[3]:(wrap_param[4] + 1), :]
            plant_seg_guas = np.concatenate((plant_seg_guas[:, 0:u_half_size] * 0, plant_seg_guas, plant_seg_guas[:, 0:u_half_size] * 0), axis=1)

            plant_seg_guas = plant_seg_guas[:, int(u_half_size + wrap_param[0] - 0.5 * (wrap_param[1] - wrap_param[0])):int(u_half_size + wrap_param[1] + 0.5 * (wrap_param[1] - wrap_param[0]))]

            plant_seg_guas = cv2.warpPerspective(plant_seg_guas, wrap_H, (plant_seg_guas.shape[1], plant_seg_guas.shape[0]))

        cv_image_lines = cv_image.copy()
        img_size = plant_seg_guas.shape

        if self.lane_det_track_img is None:
            self.lane_det_track_img = np.zeros((img_size[0], img_size[1]))

        track_on = 1
        track_img_pre_weight = 0.5
        if track_on:
            if np.max(self.lane_det_track_img)>0.01:
                plant_seg_guas = plant_seg_guas + self.lane_det_track_img * track_img_pre_weight

        plant_seg_guas_lines = np.stack(((plant_seg_guas * (255.0 / np.max(plant_seg_guas))).astype(np.uint8),) * 3, axis=-1)

        x_range = np.array([x for x in range(img_size[1])])
        y_range = np.array([x for x in range(img_size[0])])
        theta_range = np.array([theta for theta in range(-30, 30 + 1)])

        his = np.zeros((img_size[1], theta_range.shape[0]))

        # t1 = time.time()
        # print('t1 = %f' % (t1-t0))

        # handle.GrayScaleHoughLine.argtypes = [ctypes.POINTER(ctypes.c_double)]
        # res = handle.GrayScaleHoughLine(np.array([0.0, 1.0]).ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        cvar_x_range_num = int(img_size[1])
        cvar_y_range_num = int(img_size[0])
        cvar_theta_range = theta_range.astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cvar_theta_range_num = int(theta_range.shape[0])
        cvar_plant_seg_guas = plant_seg_guas.reshape((-1,)).astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # row first
        np_cvar_his = his.reshape((-1,)).astype(np.double)

        # t21 = time.time()
        # print('t21 = %f' % (t21 - t0))

        self.cfunc_handle.GrayScaleHoughLine(cvar_x_range_num, cvar_y_range_num, cvar_theta_range, cvar_theta_range_num, cvar_plant_seg_guas, np_cvar_his.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        # t22 = time.time()
        # print('t22 = %f' % (t22 - t0))

        np_cvar_his = np_cvar_his.reshape((img_size[1], theta_range.shape[0]))

        # for u_idx in range(x_range.shape[0]):
        #     u = x_range[u_idx]
        #
        #     for theta_idx in range(theta_range.shape[0]):
        #         theta = theta_range[theta_idx]
        #
        #         # x = -np.tan(np.deg2rad(theta)) * y_range + (u + np.tan(np.deg2rad(theta)) * (img_size[0]-1))
        #         x = -np.tan(np.deg2rad(theta)) * y_range + (u + np.tan(np.deg2rad(theta)) * 0)
        #         y = y_range.copy()
        #
        #         x_valid = x[(0 <= x) & (x < img_size[1])]
        #         y_valid = y[(0 <= x) & (x < img_size[1])]
        #
        #         x_valid = x_valid[(0 <= y_valid) & (y_valid < img_size[0])]
        #         y_valid = y_valid[(0 <= y_valid) & (y_valid < img_size[0])]
        #
        #         his[u_idx, theta_idx] = np.sum(plant_seg_guas[(img_size[0]-1)-y_valid.astype(int), x_valid.astype(int)])

        his = np_cvar_his.copy()

        # t2 = time.time()
        # print('t2 = %f' % (t2- t1))

        his_u = np.max(his, axis=1)

        track_u_add = his_u*0
        track_u_width = 20
        if self.lane_det_track_u is not None:
            track_u_add[int(max(0, self.lane_det_track_u[0]-track_u_width)) : int(min(self.lane_det_track_u[0]+track_u_width, img_size[1]))] = 1
            track_u_add[int(max(0, self.lane_det_track_u[1]-track_u_width)) : int(min(self.lane_det_track_u[1]+track_u_width, img_size[1]))] = 1

        # print(self.lane_det_track_u)

        track_u_weight = 1.0
        if track_on:
            his_u = his_u + track_u_add*(np.max(his_u)-np.min(his_u))*track_u_weight

        his_u_ext = np.block([np.array([0]), his_u, np.array([0])])
        peaks_ext, peaks_ext_info = signal.find_peaks(his_u_ext, distance=(img_size[1]/4.0))

        if peaks_ext.shape[0]<2:
            lines = None
            rospy.logwarn('lane_det_gaus: less than 2 lanes detected. ')
            return lines, plant_seg_guas_lines, cv_image_lines

        peaks = peaks_ext-1
        peaks_height = his_u[peaks]
        asc_idx = np.argsort(peaks_height)

        two_asc_idx = asc_idx[-2:]
        two_peaks = peaks[two_asc_idx]

        if two_peaks[0]>two_peaks[1]:
            two_peaks = np.flip(two_peaks)

        lane_u_d = two_peaks.copy()
        lane_theta_idx = np.array([np.argmax(his[lane_u_d[0], :]), np.argmax(his[lane_u_d[1], :])])
        lane_theta = theta_range[lane_theta_idx]
        lane_u_u = np.array([-np.tan(np.deg2rad(lane_theta[0]))*img_size[0] + lane_u_d[0], -np.tan(np.deg2rad(lane_theta[1]))*img_size[0] + lane_u_d[1]])

        # plant_seg_guas_lines = np.stack((plant_seg_guas,) * 3, axis=-1)
        cv2.line(plant_seg_guas_lines, (int(lane_u_d[0]), img_size[0]-1), (int(lane_u_u[0]), 0), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(plant_seg_guas_lines, (int(lane_u_d[1]), img_size[0]-1), (int(lane_u_u[1]), 0), (0, 255, 0), 1, cv2.LINE_AA)

        # cv_image_lines = cv_image.copy()
        cv2.line(cv_image_lines, (int(lane_u_d[0]), img_size[0] - 1), (int(lane_u_u[0]), 0), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(cv_image_lines, (int(lane_u_d[1]), img_size[0] - 1), (int(lane_u_u[1]), 0), (0, 255, 0), 1, cv2.LINE_AA)

        lines = np.block([[lane_u_d], [lane_u_u]])

        # mpl.use('TkAgg')
        # plt.imshow(plant_seg_guas_lines)
        # plt.show()

        # t3 = time.time()
        # print('t3 = %f' % (t3-t2))
        # print('++++++++++++')

        self.lane_det_track_img = self.lane_det_track_img * 0
        cv2.line(self.lane_det_track_img, (int(lane_u_d[0]), img_size[0] - 1), (int(lane_u_u[0]), 0), 1, 10, cv2.LINE_AA)
        cv2.line(self.lane_det_track_img, (int(lane_u_d[1]), img_size[0] - 1), (int(lane_u_u[1]), 0), 1, 10, cv2.LINE_AA)

        for n in range(5):
            self.lane_det_track_img = scipy.ndimage.uniform_filter(self.lane_det_track_img, size=20, mode='constant', cval=0)

        self.lane_det_lines = lines.copy()
        self.lane_det_track_u = self.lane_det_lines[0, :]

        return lines, plant_seg_guas_lines, cv_image_lines

    # detect farm lanes based on the plant segmentation image
    # (1) convert the perspective image to the bird eye image.
    # (2) use hough transform to estimate the theta angle by averaging top 10 thetas
    # (3) estimate r of line again with hough transform, by fixing the theta angle
    # (4) merge nearby lane_det_lines and compute v coordinates of lane_det_lines with max u coordinates
    def detect_lanes(self, plant_seg, cv_image):

        # choose proper params
        if self.cam_to_use == 0:
            cam_H = self.left_cam_H
            bird_img_size = self.left_bird_img_size
            bird_roi_perc = self.left_bird_roi_perc
        elif self.cam_to_use == 1:
            cam_H = self.right_cam_H
            bird_img_size = self.right_bird_img_size
            bird_roi_perc = self.right_bird_roi_perc
        else:
            rospy.logwarn('lane_det: unknown camera selection.')
            return

        # change binary image to

        plant_seg_rgb = np.asarray((plant_seg * 255).astype(np.uint8))

        # convert perspective image to the bird eye view image using camera homography matrix
        bird_eye_view_raw = cv2.warpPerspective(plant_seg_rgb, cam_H,
                                                (int(bird_img_size[0]), int(bird_img_size[1])), flags=cv2.INTER_NEAREST)
        bird_eye_view_rgb = cv2.warpPerspective(cv_image, cam_H,
                                                (int(bird_img_size[0]), int(bird_img_size[1])), flags=cv2.INTER_NEAREST)

        # only keep plants in roi
        detect_roi = bird_roi_perc  # u_min, u_max, v_min, v_max, in percentage
        bird_eye_view = bird_eye_view_raw.copy()
        bird_eye_view[:, :int(detect_roi[0] * bird_eye_view.shape[1])] = 0
        bird_eye_view[:, int(detect_roi[1] * bird_eye_view.shape[1]):] = 0
        bird_eye_view[:int(detect_roi[2] * bird_eye_view.shape[0]), :] = 0
        bird_eye_view[int(detect_roi[3] * bird_eye_view.shape[0]):, :] = 0

        # 1st hough transform,
        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), self.param['lane_det']['hough_hist_thr'], None, 0, 0,
                               np.deg2rad(self.param['lane_det']['hough_theta_range'][0]),
                               np.deg2rad(self.param['lane_det']['hough_theta_range'][1]))
        if lines is None:
            rospy.logwarn('HoughLines: no lane_det_lines are detected.')
            return None, None, None

        # lines2d [[r1, r2, ...], [theta1, theta2, ...]]
        lines2d = lines.reshape((lines.shape[0], lines.shape[2]))
        # for those r<0, change its theta to theta+pi, so its r can be positive
        lines2d[lines2d[:, 0] < 0, 1] = ((lines2d[lines2d[:, 0] < 0, 1] + np.pi) + np.pi) % (
                    2 * np.pi) - np.pi  # wrap to pi
        lines2d[lines2d[:, 0] < 0, 0] = -lines2d[lines2d[:, 0] < 0, 0]
        # extract the top 10 lane_det_lines
        lines_theta = lines2d[0:10, 1]
        lines_r = lines2d[0:10, 0]

        # get the average theta whose r is within a range around the middle line+
        thetas = np.zeros((10,))
        idx = 0
        for i in range(lines_theta.shape[0]):
            if (self.param['lane_det']['hough_theta_sel_r_range'][0]/self.bird_pixel_size + bird_eye_view.shape[1]/2.0) \
                    < lines_r[i] < \
                    (self.param['lane_det']['hough_theta_sel_r_range'][1]/self.bird_pixel_size + bird_eye_view.shape[1]/2.0):
                thetas[idx] = lines_theta[i]
                idx = idx + 1
        lines_theta = thetas[0:idx]
        if idx == 0:
            rospy.logwarn('detect_lanes: not enough middle lane_det_lines detected.')
            return None, None, None
        theta = np.mean(lines_theta)

        # 2nd hough transform, fix the theta and find the r
        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), self.param['lane_det']['hough_hist_thr'], None, 0, 0, \
                               theta, theta + np.deg2rad(1))
        if lines is None:
            rospy.logwarn('HoughLines: no lane_det_lines are detected.')
            return None, None, None
        lines_r = lines.reshape((lines.shape[0], lines.shape[2]))[:, 0]

        # combine detected lane_det_lines close to half lane width
        r_thr = self.max_lane_width * 0.5 / self.bird_pixel_size
        # order r in ascending order, r can be both positive or negative now
        lines_r_asc = np.sort(lines_r)
        # prepare var to save the combined lane params (r, theta)
        lines_r_sel = np.zeros((2, lines_r_asc.shape[0]))
        lines_r_sel[1, :] = theta
        # loop around and lanes and combine lanes close to each other, and average their r
        idx = 0
        n = 0
        for i in range(lines_r_asc.shape[0]):
            r = lines_r_asc[i]
            if i == 0:
                lines_r_sel[0, idx] = r
                n = 1
            else:
                if abs(r - lines_r_asc[i - 1]) < r_thr:
                    lines_r_sel[0, idx] = lines_r_sel[0, idx] + r
                    n = n + 1
                else:
                    lines_r_sel[0, idx] = lines_r_sel[0, idx] / n

                    idx = idx + 1
                    lines_r_sel[0, idx] = lines_r_sel[0, idx] + r
                    n = 1
        # don't forget the average the r of the last combined lane
        lines_r_sel[0, idx] = lines_r_sel[0, idx] / n
        # finish extracting all lanes
        lines_r_sel = lines_r_sel[:, 0:idx + 1]

        # prepare var of bird eye image with lanes for plotting, change the image from [height, width] to
        # [height, width, channel]
        bird_eye_view_lines = np.stack((bird_eye_view,) * 3, axis=-1)
        # prepare var of lanes params consist of x (i.e. u) and theta,
        # x is the intersection of lanes with bottom of image
        lines_x_theta = None
        if lines_r_sel is not None:
            lines_x_theta = lines_r_sel.copy()
            for i in range(lines_r_sel.shape[1]):
                rho = lines_r_sel[0, i]
                theta = lines_r_sel[1, i]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(bird_eye_view_lines, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(bird_eye_view_rgb, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
                pt_edge_x = (rho - bird_eye_view.shape[0] * b) / a
                lines_x_theta[0, i] = pt_edge_x
                cv2.circle(bird_eye_view_lines, (int(bird_eye_view.shape[1] / 2.0), bird_eye_view.shape[0]), 5,
                           (255, 0, 0), 1)
                cv2.circle(bird_eye_view_lines, (int(pt_edge_x), bird_eye_view.shape[0]), 5, (0, 0, 255), 1)

        # return the lane params, and bird eye image with lanes
        return lines_x_theta, bird_eye_view_lines , bird_eye_view_rgb

    # compute weeder's absolute position shift based on the detected lane params.
    # lane_det_lines: 2XN matrix with [[x1, x2, ...],[theta1, theta2, ...]]
    # where x1, x2, ... are u coordinates of intersections of lanes with the bottom of the image
    def control_weeder(self, lines):
        # compute lanes' closest y coords in the weeder frame from their u coordinates of intersections of lanes with
        # the bottom of the image
        lines_u_pix = lines[0, :]
        lines_y = -(lines_u_pix - (self.left_bird_img_size[0] / 2.0)) * self.bird_pixel_size + self.control_bias
        # make sure there are lane both on left and right sides
        if (lines_y[lines_y > 0].shape[0] == 0) or (lines_y[lines_y < 0].shape[0] == 0):
            rospy.logwarn('no lane at left and right side is detected.')
            return

        # find the closest lanes at the left and right side, extract their y coords
        pos_idx = np.argmin(abs(lines_y[lines_y > 0] - 0))
        pos_y = (lines_y[lines_y > 0])[pos_idx]
        neg_idx = np.argmin(abs(lines_y[lines_y < 0] - 0))
        neg_y = (lines_y[lines_y < 0])[neg_idx]

        # make sure the closest left and right lanes are within max_lane_width,
        # if so, find the middle of the two lanes by averaging their y
        if pos_y < self.max_lane_width and neg_y > -self.max_lane_width:
            mid_y = (pos_y + neg_y) / 2.0
        else:
            rospy.logwarn('lane y exceed the max lane width')
            return

        # after a fixed distance, send weeder control cmd
        if self.ctl_time_pre == -1:
            self.ctl_time_pre = rospy.get_time()
        cur_time = rospy.get_time()
        if (cur_time - self.ctl_time_pre) * self.weeder_speed > self.param['weeder']['cmd_dist']:
            step_num = int(np.floor((cur_time - self.ctl_time_pre) * self.weeder_speed / self.param['weeder']['cmd_dist']))  # TODO
            for step in range(step_num):  # TODO
                self.mid_y_buff[0:self.mid_y_buff.shape[0] - 1] = self.mid_y_buff[1:self.mid_y_buff.shape[0]].copy()  # TODO
                self.mid_y_buff[self.mid_y_buff.shape[0] - 1] = mid_y  # TODO
                self.weeder_pos_buff[0:self.weeder_pos_buff.shape[0] - 1] = self.weeder_pos_buff[1:self.weeder_pos_buff.shape[0]].copy()  # TODO
                self.weeder_pos_buff[self.weeder_pos_buff.shape[0] - 1] = self.weeder_pos  # TODO

            self.ctl_time_pre = cur_time

            # self.weeder_y = self.mid_y_buff[0]  # extract the earliest position shift
            if self.weeder_cmd_delay:  # TODO
                self.weeder_y = self.mid_y_buff[0] + self.weeder_pos_buff[0] - self.weeder_pos  # TODO
            else:  # TODO
                self.weeder_y = self.mid_y_buff[-1]  # extract the most recent position shift  # TODO

            # compute the control command u using the classic P controller
            u = self.weeder_y * self.param['weeder']['ctrl_p']  # p controller
            if abs(u) < self.param['weeder']['ctrl_u_min']:
                u = np.sign(u) * self.param['weeder']['ctrl_u_min']
            # add control cmd u to the previous weeder position
            self.ctl_cmd = self.ctl_cmd + u

            # send the control cmd, i.e. the absolute position shift of the weeder
            msg = Float32MultiArray()
            msg.data = [self.ctl_cmd]
            self.weeder_control_pub.publish(msg)
            if self.verbose:
                rospy.loginfo('ctl_cmd = %f \n' % (self.ctl_cmd))
        else:
            return

        pass

    def control_weeder_gaus(self, lines):
        wrap_param = self.wrap_param

        lines_u_d = lines[0, :]
        lane_u_mid = (lines_u_d[0] + lines_u_d[1])/2.0

        weeder_pos_pix = wrap_param[5]-wrap_param[0]+int((wrap_param[1]-wrap_param[0])*0.5)

        lane_u_offset = lane_u_mid-(self.wrap_img_size[1]/2.0)

        lane_y_offset = -lane_u_offset*(0.4/(lines_u_d[1]-lines_u_d[0]))

        mid_y = lane_y_offset

        # after a fixed distance, send weeder control cmd
        if self.ctl_time_pre == -1:
            self.ctl_time_pre = rospy.get_time()
        cur_time = rospy.get_time()
        if (cur_time - self.ctl_time_pre) * self.weeder_speed > self.param['weeder']['cmd_dist']:
            step_num = int(np.floor((cur_time - self.ctl_time_pre) * self.weeder_speed / self.param['weeder']['cmd_dist']))  # TODO
            for step in range(step_num):  # TODO
                self.mid_y_buff[0:self.mid_y_buff.shape[0] - 1] = self.mid_y_buff[1:self.mid_y_buff.shape[0]].copy()  # TODO
                self.mid_y_buff[self.mid_y_buff.shape[0] - 1] = mid_y  # TODO
                self.weeder_pos_buff[0:self.weeder_pos_buff.shape[0] - 1] = self.weeder_pos_buff[1:self.weeder_pos_buff.shape[0]].copy()  # TODO
                self.weeder_pos_buff[self.weeder_pos_buff.shape[0] - 1] = self.weeder_pos  # TODO

            self.ctl_time_pre = cur_time

            # self.weeder_y = self.mid_y_buff[0]  # extract the earliest position shift
            if self.weeder_cmd_delay:  # TODO
                self.weeder_y = self.mid_y_buff[0] + self.weeder_pos_buff[0] - self.weeder_pos  # TODO
            else:  # TODO
                self.weeder_y = self.mid_y_buff[-1]  # extract the most recent position shift  # TODO

            # send the control cmd, i.e. the absolute position shift of the weeder
            msg = Float32MultiArray()
            msg.data = [self.weeder_y]
            self.weeder_control_pub.publish(msg)
            print('weeder cmd: ' % self.weeder_y)
            if self.verbose:
                rospy.loginfo('ctl_cmd = %f \n' % (self.weeder_y))
        else:
            return

        pass

    # compute homography based on camera's intrinsic and extrinsic matrix, K and T
    # K: camera intrinsic matrix
    # T: camera extrinsic, camera pose in ground coord. the ground coord is (w.r.t. the weeder):
    # X to the right, Y to the back, Z to the down
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
        # # change the order to (v,u) to be consistent with image size
        # bird_img_size = np.array([bird_img_size[1], bird_img_size[0]])

        # compute homography matrix
        H = cv2.getPerspectiveTransform(inputpts, outputpts)

        # return vars
        return bird_img_size, H

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


def main(args):
    rospy.init_node('image_processor_node', anonymous=True)
    img_proc = image_processor()

    # if camera selection is failed, return
    if img_proc.cam_to_use==-1:
        rospy.logwarn('lane_det: camera selection failed, ending ...')
        return False

    # the main loop for line detection
    while not rospy.is_shutdown():
        img_proc.img_cb()


if __name__ == '__main__':
    main(sys.argv)
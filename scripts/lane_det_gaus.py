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

import torch

import rospkg
import yaml

import time
import ctypes

mpl.use('TkAgg')
# plt.imshow(image_seg_bn, cmap='gray')
# plt.show()

class lane_detector:
    def __init__(self):
        # locate ros pkg
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'

        # load key params
        params_filename = rospy.get_param('param_file')  # self.pkg_path + 'cfg/' + 'param.yaml'
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)
        self.verbose = self.param['verbose']
        self.img_size = np.array([self.param['cam']['height'], self.param['cam']['width']])

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
        # self.det_img_left_pub = rospy.Publisher('/det_img/image_raw_left', Image, queue_size=2)
        # self.det_img_right_pub = rospy.Publisher('/det_img/image_raw_right', Image, queue_size=2)
        self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=2)
        self.seg_line_pub = rospy.Publisher('/seg_line/image_raw', Image, queue_size=2)
        # self.line_rgb_left_pub = rospy.Publisher('/line_rgb/image_raw_left', Image, queue_size=2)
        # self.line_rgb_right_pub = rospy.Publisher('/line_rgb/image_raw_right', Image, queue_size=2)
        self.rgb_line_pub = rospy.Publisher('/rgb_line/image_raw', Image, queue_size=2)
        # get camera intrinsic
        self.left_cam_K = np.array(self.param['cam']['left']['K']).reshape((3, 3))
        self.right_cam_K = np.array(self.param['cam']['right']['K']).reshape((3, 3))
        # get camera extrinsic
        self.left_cam_T = transform_trans_ypr_to_matrix(np.array(self.param['cam']['left']['T']).reshape((6, 1)))
        self.right_cam_T = transform_trans_ypr_to_matrix(np.array(self.param['cam']['right']['T']).reshape((6, 1)))

        # lane det params
        self.wrap_img_on = self.param['lane_det']['wrap_img_on']
        self.wrap_img_size_u = self.param['lane_det']['wrap_img_size_u']
        # wrap_param: [left lane bot pix u, right lane bot pix u, left lane top pix u, right lane top pix u, bot v, top v, weeder bot u]
        self.wrap_param_left_cam = np.array(self.param['lane_det']['wrap_param_left_cam']).reshape((-1,))
        self.wrap_param_right_cam = np.array(self.param['lane_det']['wrap_param_right_cam']).reshape((-1,))
        self.wrap_H_left_cam, self.wrap_H_right_cam = self.compute_wrap_H(self.wrap_param_left_cam, self.wrap_param_right_cam)
        self.wrap_param = None  # dynamically change according to left and right cam selection
        self.wrap_img_size = None
        self.wrap_H = None  # dynamically change according to left and right cam selection
        self.resize_proportion = None # proportion of resize operation in image wrapping

        self.cfunc_handle = ctypes.CDLL(self.pkg_path + "libGrayScaleHoughLine.so") # cpp function call
        self.lane_det_track_img = None
        self.lane_det_track_u = None
        self.lane_det_lines = None

        # weeder pub, sub and vars
        self.weeder_speed = self.param['weeder']['def_speed']
        self.weeder_pos = self.param['weeder']['def_pos']
        self.weeder_status = self.param['weeder']['def_status']
        self.weeder_control_pub = rospy.Publisher('/weeder_cmd', Float32MultiArray, queue_size=2)
        self.weeder_speed_pos_sub = rospy.Subscriber('/weeder_speed_pos', Float32MultiArray, self.weeder_speed_pos_cb)
        self.weeder_cmd = 0
        if self.param['weeder']['cmd_dist'] == -1:
            self.weeder_cmd_buff = np.zeros((int(self.param['weeder']['cam_weeder_dist'] / self.param['weeder']['cmd_dist_def']), ))
        else:
            self.weeder_cmd_buff = np.zeros((int(self.param['weeder']['cam_weeder_dist']/self.param['weeder']['cmd_dist']),))
        self.ctl_time_pre = rospy.get_time()
        self.weeder_cmd_delay = self.param['weeder']['weeder_cmd_delay']
        self.lane_y_offset = 0.0

        # logging
        self.log_on = self.param['log']['enable']
        self.log_msg_pub = rospy.Publisher('/log_msg', String, queue_size=2)

        # # TODO: TEST
        # self.test_t = rospy.get_time()

        return

    # compute left and right camera's wrap H matrix
    def compute_wrap_H(self, wrap_param_left_cam, wrap_param_right_cam):
        # compute for the left cam
        wrap_param = wrap_param_left_cam
        wrap_img_crop_size = np.array([int(wrap_param[4] - wrap_param[5] + 1), int(2 * (wrap_param[1] - wrap_param[0]))])
        h_shift = (wrap_param[0] + wrap_param[1]) / 2.0 - (wrap_param[2] + wrap_param[3]) / 2.0
        if h_shift > 0:
            inputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1] - int(h_shift), 0], [0, 0]])
            outputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1], 0], [int(h_shift), 0]])
        else:
            inputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1], 0], [int(abs(h_shift)), 0]])
            outputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1] - int(abs(h_shift)), 0], [0, 0]])
        wrap_H_left_cam = cv2.getPerspectiveTransform(inputpts, outputpts)
        # compute for the right cam
        wrap_param = wrap_param_right_cam
        wrap_img_crop_size = np.array([int(wrap_param[4] - wrap_param[5] + 1), int(2 * (wrap_param[1] - wrap_param[0]))])
        h_shift = (wrap_param[0] + wrap_param[1]) / 2.0 - (wrap_param[2] + wrap_param[3]) / 2.0
        if h_shift > 0:
            inputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1] - int(h_shift), 0], [0, 0]])
            outputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1], 0], [int(h_shift), 0]])
        else:
            inputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1], 0], [int(abs(h_shift)), 0]])
            outputpts = np.float32([[0, wrap_img_crop_size[0]], [wrap_img_crop_size[1], wrap_img_crop_size[0]],[wrap_img_crop_size[1] - int(abs(h_shift)), 0], [0, 0]])
        wrap_H_right_cam = cv2.getPerspectiveTransform(inputpts, outputpts)
        # return wrap_H matrix
        return wrap_H_left_cam, wrap_H_right_cam

    # sub for weeder weeder speed and pos, simply store it in member var
    def weeder_speed_pos_cb(self, msg):
        self.weeder_speed = msg.data[0]
        self.weeder_pos = msg.data[1]
        self.weeder_status = msg.data[2]

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
            rospy.logwarn('lane_det: not enough plant area detected when selecting camera. using camera 1')
            self.cam_to_use = 0
            # return False

        return True

    # the main loop: detect line and send back the result
    def img_cb(self):
        log_msg = String()
        t0 = rospy.get_time()
        if self.log_on:
            log_msg.data = str(rospy.get_time()) + ': start processing a new image.'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)

        # (0) choose camera in each cam_sel_time_interval time
        choose_cam_cur_time = rospy.get_time()
        if choose_cam_cur_time - self.choose_cam_pre_time >= self.cam_sel_time_interval:
            self.select_cam()
            self.lane_det_track_img = None  # re-initialize tracking of the lane det result
            self.choose_cam_pre_time = rospy.get_time()

            t01 = rospy.get_time()
            if self.log_on:
                log_msg.data = str(t01) + ': finished camera selection process.'
                self.log_msg_pub.publish(log_msg)
                rospy.sleep(0.001)

        # # TODO: TEST
        # t = rospy.get_time()
        # if t - self.test_t > 5:
        #     if self.cam_to_use == 0:
        #         self.cam_to_use = 1
        #     else:
        #         self.cam_to_use = 0
        #     self.lane_det_track_img = None
        #     self.test_t = t
        #     rospy.logwarn('changed cam to cam: ' + str(self.cam_to_use))

        # select left or right camera image, and their corresponding params for lane detection
        if self.cam_to_use == 0:
            cv_image = self.left_img
            self.wrap_param = self.wrap_param_left_cam
            self.wrap_H = self.wrap_H_left_cam
            if self.log_on:
                log_msg.data = str(rospy.get_time()) + ': using left camera image.'
                self.log_msg_pub.publish(log_msg)
                rospy.sleep(0.001)
        elif self.cam_to_use == 1:
            cv_image = self.right_img
            self.wrap_param = self.wrap_param_right_cam
            self.wrap_H = self.wrap_H_right_cam
            if self.log_on:
                log_msg.data = str(rospy.get_time()) + ': using right camera image.'
                self.log_msg_pub.publish(log_msg)
                rospy.sleep(0.001)
        else:
            rospy.logwarn('lane_det: unknown camera selection.')
            return
        self.wrap_img_size = np.array([int((self.wrap_param[4] - self.wrap_param[5] + 1) *
                                           (self.wrap_img_size_u / (2 * (self.wrap_param[1] - self.wrap_param[0])))), self.wrap_img_size_u])
        self.resize_proportion = (self.wrap_img_size_u / ((self.wrap_param[1] - self.wrap_param[0]) * 2.0))

        if cv_image is None:
            if self.log_on:
                log_msg.data = str(rospy.get_time()) + ': camera image not ready, skipping.'
                self.log_msg_pub.publish(log_msg)
                rospy.sleep(0.001)
            rospy.sleep(0.5)
            return
        else:
            # publish image to be processed
            try:
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            except CvBridgeError as e:
                print(e)
                return
            self.proc_img_pub.publish(ros_image)

        t02 = rospy.get_time()
        if self.verbose:
            rospy.loginfo('time consumption until preparing image = %f' % (t02 - t0))
        if self.log_on:
            log_msg.data = str(rospy.get_time()) + ': finished preparing image.'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t02 - t0)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)

        # (1) segment plants in the image
        # segment plants with yolov5
        plant_seg_bn, det_image, plant_seg_gaus = self.segment_plants_yolo5(cv_image)

        # publish segmentation results
        plant_seg_pub = np.asarray((plant_seg_bn * 255).astype(np.uint8))
        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_pub, "mono8")
            det_image = self.bridge.cv2_to_imgmsg(det_image,"bgr8")
        except CvBridgeError as e:
            print(e)
            return
        self.seg_img_pub.publish(ros_image)
        # if self.cam_to_use==0:
        #     self.det_img_left_pub.publish(det_image)
        # elif self.cam_to_use==1:
        #     self.det_img_right_pub.publish(det_image)
        self.det_img_pub.publish(det_image)

        t1 = rospy.get_time()
        if self.verbose:
            rospy.loginfo('time consumption until yolo5 based image det and seg = %f' % (t1 - t0))
        if self.log_on:
            log_msg.data = str(rospy.get_time()) + ': finished yolo5 based image det and seg.'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t1 - t0)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)

        # (2) detect lanes
        # extract lane_det_lines with the two step hough transform method
        lines, seg_lines, rgb_lines = self.detect_lanes(plant_seg_gaus, cv_image)
        if lines is None:
            rospy.logwarn('no lane_det_lines are detected, skipping this image and returning ...')
            return

        # publish line detection results
        try:
            ros_seg_lines_image = self.bridge.cv2_to_imgmsg(seg_lines, "bgr8")
            ros_rgb_lines_image = self.bridge.cv2_to_imgmsg(rgb_lines, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        self.seg_line_pub.publish(ros_seg_lines_image)
        # if self.cam_to_use == 0:
        #     self.line_rgb_left_pub.publish(rgb_lines)
        # elif self.cam_to_use==1:
        #     self.line_rgb_right_pub.publish(rgb_lines)
        self.rgb_line_pub.publish(ros_rgb_lines_image)

        t2 = rospy.get_time()
        if self.verbose:
            rospy.loginfo('time consumption until line det = %f' % (t2 - t0))
        if self.log_on:
            log_msg.data = str(rospy.get_time()) + ': finished line det.'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t2 - t0)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)

        # (3) control weeder
        # compute and send weeder cmd
        weeder_cmd = self.control_weeder(lines)
        if self.log_on:
            log_msg.data = str(rospy.get_time()) + ': current weeder_pos = ' + str(self.weeder_pos)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': current weeder_speed = ' + str(self.weeder_speed)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': lane center Y offset = ' + str(self.lane_y_offset)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            if weeder_cmd is not None:
                log_msg.data = str(rospy.get_time()) + ': new lane det weeder_cmd = ' +str(weeder_cmd)
                self.log_msg_pub.publish(log_msg)
                rospy.sleep(0.001)
            else:
                log_msg.data = str(rospy.get_time()) + ': no new lane det weeder_cmd sent. '
                self.log_msg_pub.publish(log_msg)
                rospy.sleep(0.001)

        t3 = rospy.get_time()
        if self.verbose:
            rospy.loginfo('time consumption until weeder ctl = %f' % (t3 - t0))
            rospy.loginfo('++++++++++++')
        if self.log_on:
            log_msg.data = str(rospy.get_time()) + ': finished sending weeder ctl.'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': time consumption until now = ' + str(t3 - t0)
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)
            log_msg.data = str(rospy.get_time()) + ': ++++++++++++'
            self.log_msg_pub.publish(log_msg)
            rospy.sleep(0.001)

        # return once everything successes
        return

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
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            if int(cls) == 0 and conf > 0.4 and y2 - y1 < 15 and x2 - x1 < 15:  # only select id=0, i.e. plants
                cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            if int(cls) == self.yolo5_plant_id:  # only select id=0, i.e. plants
                # image_seg_bn = cv2.rectangle(image_seg_bn, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1) # draw rect
                image_seg_bn = cv2.circle(image_seg_bn, (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
                                          int(np.min(np.array([x2 - x1, y2 - y1]) / 2.0)), 1, -1)  # draw circle
                cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # ma blob img
        # ma_det = scipy.ndimage.uniform_filter(image_seg_bn, size=3, axes=None)
        img_seg_ma = image_seg_bn.copy()
        for n in range(3):
            # img_seg_ma = scipy.ndimage.uniform_filter(img_seg_ma, size=10)  # don't use scipy.ndimage.uniform_filter, it will shift the location slightly
            img_seg_ma = cv2.GaussianBlur(img_seg_ma, (11, 11), 0)


        # return seg result, i.e. a binary image
        return image_seg_bn, det_image, img_seg_ma

    def detect_lanes(self, plant_seg_guas, cv_image):
        # wrap the raw image to let the middle lane in the middle of the image and vertically straight up, if needed.
        if self.wrap_img_on:
            wrap_param = self.wrap_param

            # fill two sides of the image will black image with half of its width
            u_half_size = int(self.img_size[1] / 2.0)
            cv_image = np.concatenate((cv_image[:, 0:u_half_size, :] * 0, cv_image, cv_image[:, 0:u_half_size, :] * 0), axis=1)
            plant_seg_guas = np.concatenate((plant_seg_guas[:, 0:u_half_size] * 0, plant_seg_guas, plant_seg_guas[:, 0:u_half_size] * 0), axis=1)
            # crop image, 1st vertical then 2nd horizontal
            cv_image = cv_image[wrap_param[5]:(wrap_param[4] + 1), :, :]  # vertical crop
            cv_image = cv_image[:, int(u_half_size + wrap_param[0] - 0.5 * (wrap_param[1] - wrap_param[0])):
                                   int(u_half_size + wrap_param[1] + 0.5 * (wrap_param[1] - wrap_param[0])), :]
            plant_seg_guas = plant_seg_guas[wrap_param[5]:(wrap_param[4] + 1), :]  # horizontal crop
            plant_seg_guas = plant_seg_guas[:, int(u_half_size + wrap_param[0] - 0.5 * (wrap_param[1] - wrap_param[0])):
                                               int(u_half_size + wrap_param[1] + 0.5 * (wrap_param[1] - wrap_param[0]))]
            # transform
            cv_image = cv2.warpPerspective(cv_image, self.wrap_H, (cv_image.shape[1], cv_image.shape[0]))
            plant_seg_guas = cv2.warpPerspective(plant_seg_guas, self.wrap_H, (plant_seg_guas.shape[1], plant_seg_guas.shape[0]))

            # resize
            cv_image_bf_resize = cv_image.copy()  # save img before resize
            plant_seg_guas_bf_resize = plant_seg_guas.copy()  # save img before resize
            cv_image = cv2.resize(cv_image, (int(self.wrap_img_size_u), int(cv_image.shape[0] * (self.wrap_img_size_u / cv_image.shape[1]))))
            plant_seg_guas = cv2.resize(plant_seg_guas, (int(self.wrap_img_size_u), int(plant_seg_guas.shape[0] * (self.wrap_img_size_u / plant_seg_guas.shape[1]))))
        else:
            cv_image_bf_resize = cv_image.copy()  # save img before resize

        img_size = plant_seg_guas.shape  # image size after wrapping

        # track previous line det results
        if self.lane_det_track_img is None:
            self.lane_det_track_img = np.zeros((img_size[0], img_size[1]))
        track_on = self.param['lane_det']['track_on']
        track_img_pre_weight = self.param['lane_det']['track_img_pre_weight']
        if track_on:
            if np.max(self.lane_det_track_img)>0.01:
                plant_seg_guas = plant_seg_guas + self.lane_det_track_img * track_img_pre_weight  # TODO
                plant_seg_guas[plant_seg_guas>1] = 1

        # prepare vars for searching best two parallel lines
        x_range = np.array([x for x in range(int(img_size[1]/2.0))])  # u ranges in wrapped image
        y_range = np.array([x for x in range(img_size[0])])  # v ranges in wrapped image
        # left line's theta angle from wrap_param, i.e. to find ref's left line theta
        ll_theta_from_wrap_param = -np.rad2deg(np.arctan2(((self.wrap_param[1]-self.wrap_param[0])/2.0) - ((self.wrap_param[3]-self.wrap_param[2])/2.0),
                                                          (self.wrap_param[4] - self.wrap_param[5])))
        theta_range = np.array([theta for theta in range(-self.param['lane_det']['theta_range'], self.param['lane_det']['theta_range'] + 1)]) \
                      + ll_theta_from_wrap_param  # theta range to search from

        # ref left and right lines' intersection point's v coord dist from the bottom of the image
        inter_h = ((self.wrap_param[1]-self.wrap_param[0]) / ((self.wrap_param[1]-self.wrap_param[0]) - (self.wrap_param[3]-self.wrap_param[2]))) * self.wrap_img_size[0]
        shift_range = np.array([x for x in range(-self.param['lane_det']['shift_range'], self.param['lane_det']['theta_range'] + 1)])

        # construct 3D hist for search best left and right lines,
        # his_3d: [search left line's bot u, left line's theta, right line's shift from left line's bot u + ref lane widht]
        his_3d = np.zeros((x_range.shape[0], theta_range.shape[0], shift_range.shape[0]))

        # prepare vars for C++ based hist computation operation
        cvar_img_size_u = int(self.wrap_img_size[1])
        cvar_img_size_v = int(self.wrap_img_size[0])
        cvar_x_range_num = int(x_range.shape[0])
        cvar_y_range_num = int(y_range.shape[0])
        cvar_theta_range = theta_range.astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cvar_theta_range_num = int(theta_range.shape[0])
        cvar_shift_range = shift_range.astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cvar_shift_range_num = int(shift_range.shape[0])
        cvar_inter_h = ctypes.c_double(inter_h)
        cvar_plant_seg_guas = plant_seg_guas.reshape((-1,)).astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # row first
        np_cvar_his = his_3d.reshape((-1,)).astype(np.double)

        # compute hist 3D: np_cvar_his, using C++
        self.cfunc_handle.GrayScaleHoughLine(cvar_img_size_u, cvar_img_size_v, cvar_x_range_num, cvar_y_range_num, cvar_theta_range, cvar_theta_range_num,
                                             cvar_shift_range, cvar_shift_range_num, cvar_inter_h,
                                             cvar_plant_seg_guas, np_cvar_his.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        his_3d = np_cvar_his.reshape((x_range.shape[0], theta_range.shape[0], shift_range.shape[0])).copy()  # recover its 3D shape

        # ref lines' bot (down) pixel's  and top (up) pixel's u
        ref_lines_u_d = np.array([self.wrap_img_size_u*0.25, self.wrap_img_size_u*0.75])
        ref_lines_u_u = np.array([self.wrap_img_size_u*0.5 - ((self.wrap_param[3]-self.wrap_param[2])/2.0)*(self.wrap_img_size_u/((self.wrap_param[1]-self.wrap_param[0])*2.0)),
                                  self.wrap_img_size_u*0.5 + ((self.wrap_param[3]-self.wrap_param[2])/2.0)*(self.wrap_img_size_u/((self.wrap_param[1]-self.wrap_param[0])*2.0))])
        ref_lines_theta = np.array([np.rad2deg(np.arctan2(ref_lines_u_d[0] - ref_lines_u_u[0], self.wrap_img_size[0])),
                                    np.rad2deg(np.arctan2(ref_lines_u_d[1] - ref_lines_u_u[1], self.wrap_img_size[0]))])
        ref_lines_idx = np.array([np.argmin(np.abs(x_range - ref_lines_u_d[0])), np.argmin(np.abs(theta_range - ref_lines_theta[0])), (shift_range.shape[0]-1)/2.0])
        # ref lines' heuristic score
        ref_lines_heur = his_3d[int(ref_lines_idx[0]), int(ref_lines_idx[1]), int(ref_lines_idx[2])]

        # from previous line det results, only search left line's bot u from the local region of the previous det result
        his_3d_u_track = his_3d.copy() * 0
        track_u_width = self.param['lane_det']['track_u_width']
        if self.lane_det_track_u is not None:
            his_3d_u_track[int(max(0, self.lane_det_track_u[0]-track_u_width)) : int(min(self.lane_det_track_u[0]+track_u_width, img_size[1])), :, :] = \
                his_3d[int(max(0, self.lane_det_track_u[0]-track_u_width)) : int(min(self.lane_det_track_u[0]+track_u_width, img_size[1])), :, :].copy()
        else:
            his_3d_u_track = his_3d.copy()
        # get the best two lines
        max_idx = np.unravel_index(np.argmax(his_3d_u_track), his_3d_u_track.shape)
        track_lines_heur = his_3d[max_idx[0], max_idx[1], max_idx[2]]

        if track_lines_heur>ref_lines_heur:
            # collect left and right lines' params
            ll_u = x_range[max_idx[0]]  # left line's bot u
            ll_theta = theta_range[max_idx[1]]  # left line's theta
            rl_u = ll_u + self.wrap_img_size[1]/2.0 + shift_range[max_idx[2]]  # right line's bot u
            inter_u = ll_u + inter_h * np.tan(-np.deg2rad(ll_theta))  # left and right lines intersection point's u coord
            rl_theta = np.rad2deg(np.arctan2(rl_u - inter_u, inter_h))  # right line's theta

            lane_u_d = np.array([ll_u, rl_u])  # two lines' bottom points' u coord
            lane_theta = np.array([ll_theta, rl_theta])  # two lines' theta
            # two lines' upper points' u coord
            lane_u_u = np.array([-np.tan(np.deg2rad(lane_theta[0]))*img_size[0] + lane_u_d[0], -np.tan(np.deg2rad(lane_theta[1]))*img_size[0] + lane_u_d[1]])
        else:
            lane_u_d = ref_lines_u_d.copy()  # two lines' bottom points' u coord
            lane_theta = ref_lines_theta.copy()  # two lines' theta
            lane_u_u = ref_lines_u_u.copy()  # two lines' upper points' u coord

        if self.weeder_status==0:  # if weeder is up, don't detect lines
            lane_u_d = ref_lines_u_d.copy()  # two lines' bottom points' u coord
            lane_theta = ref_lines_theta.copy()  # two lines' theta
            lane_u_u = ref_lines_u_u.copy()  # two lines' upper points' u coord


        # plant seg + lines image
        plant_seg_guas_lines = np.stack(((plant_seg_guas * 255.0).astype(np.uint8),) * 3, axis=-1)
        # plant_seg_guas_lines = np.stack((plant_seg_guas,) * 3, axis=-1)
        cv2.line(plant_seg_guas_lines, (int(lane_u_d[0]), img_size[0]-1), (int(lane_u_u[0]), 0), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(plant_seg_guas_lines, (int(lane_u_d[1]), img_size[0]-1), (int(lane_u_u[1]), 0), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(plant_seg_guas_lines, (int(ref_lines_u_d[0]), img_size[0]-1), (int(ref_lines_u_u[0]), 0), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(plant_seg_guas_lines, (int(ref_lines_u_d[1]), img_size[0]-1), (int(ref_lines_u_u[1]), 0), (255, 0, 0), 1, cv2.LINE_AA)

        # raw rgb image + lines image
        # cv_image_lines = cv_image.copy()
        # cv2.line(cv_image_lines, (int(lane_u_d[0]), img_size[0] - 1), (int(lane_u_u[0]), 0), (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.line(cv_image_lines, (int(lane_u_d[1]), img_size[0] - 1), (int(lane_u_u[1]), 0), (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.line(cv_image_lines, (int(ref_lines_u_d[0]), img_size[0]-1), (int(ref_lines_u_u[0]), 0), (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.line(cv_image_lines, (int(ref_lines_u_d[1]), img_size[0]-1), (int(ref_lines_u_u[1]), 0), (255, 0, 0), 1, cv2.LINE_AA)
        cv_image_lines = cv_image_bf_resize.copy()  # show rgb image with better resolution
        cv2.line(cv_image_lines, (int(lane_u_d[0]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
                 (int(lane_u_u[0]*(1.0/self.resize_proportion)), 0), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(cv_image_lines, (int(lane_u_d[1]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
                 (int(lane_u_u[1]*(1.0/self.resize_proportion)), 0), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(cv_image_lines, (int(ref_lines_u_d[0]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
                 (int(ref_lines_u_u[0]*(1.0/self.resize_proportion)), 0), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(cv_image_lines, (int(ref_lines_u_d[1]*(1.0/self.resize_proportion)), int((img_size[0]-1)*(1.0/self.resize_proportion))),
                 (int(ref_lines_u_u[1]*(1.0/self.resize_proportion)), 0), (255, 0, 0), 1, cv2.LINE_AA)

        lines = np.block([[lane_u_d], [lane_u_u]]) # down pixel's  and up pixel's u for a line

        # re-generate tracking image, i.e. add two white lines on the black background image
        self.lane_det_track_img = self.lane_det_track_img * 0
        cv2.line(self.lane_det_track_img, (int(lane_u_d[0]), img_size[0] - 1), (int(lane_u_u[0]), 0), 1, self.param['lane_det']['track_line_width'], cv2.LINE_AA)
        cv2.line(self.lane_det_track_img, (int(lane_u_d[1]), img_size[0] - 1), (int(lane_u_u[1]), 0), 1, self.param['lane_det']['track_line_width'], cv2.LINE_AA)
        # blur the two white lines, don't use scipy.ndimage.uniform_filter or cv2.blur, they will shift the position of the line
        for n in range(5):
            self.lane_det_track_img = cv2.GaussianBlur(self.lane_det_track_img, (self.param['lane_det']['track_line_filter_width'], self.param['lane_det']['track_line_filter_width']), 0)

        # save relevant vars
        self.lane_det_lines = lines.copy()
        self.lane_det_track_u = self.lane_det_lines[0, :]

        # return results
        return lines, plant_seg_guas_lines, cv_image_lines  # line: 2x1, down pixel's  and up pixel's u for a line

    # weeder control
    def control_weeder(self, lines):
        wrap_param = self.wrap_param

        lines_u_d = lines[0, :]
        lane_u_mid = (lines_u_d[0] + lines_u_d[1])/2.0

        weeder_pos_pix = (wrap_param[6]-wrap_param[0]+int((wrap_param[1]-wrap_param[0])*0.5)) * (self.wrap_img_size_u/((wrap_param[1]-wrap_param[0])*2))
        print('weeder_pos_pix = %f' % weeder_pos_pix)

        lane_u_offset = lane_u_mid - weeder_pos_pix
        print('lane_u_offset = %f' % lane_u_offset)
        # lane_u_offset = lane_u_mid-(self.wrap_img_size_u / 2.0)

        # y coord of the machine is pointing to left
        if self.cam_to_use == 0:
            lane_y_offset = -lane_u_offset*(self.param['weeder']['farm_lane_dist_left_cam']/(lines_u_d[1]-lines_u_d[0]))
        elif self.cam_to_use == 1:
            lane_y_offset = -lane_u_offset*(self.param['weeder']['farm_lane_dist_right_cam']/(lines_u_d[1]-lines_u_d[0]))
        else:
            rospy.logwarn('lane_det: unknown camera selection.')
            return

        # apply max shift
        if lane_y_offset > self.param['weeder']['weeder_cmd_max_shift']:
            lane_y_offset = self.param['weeder']['weeder_cmd_max_shift']
        elif lane_y_offset < (-self.param['weeder']['weeder_cmd_max_shift']):
            lane_y_offset = -self.param['weeder']['weeder_cmd_max_shift']

        self.lane_y_offset = lane_y_offset
        weeder_cmd = lane_y_offset + self.weeder_pos

        # after a fixed distance, send weeder control cmd
        cur_time = rospy.get_time()
        if (cur_time - self.ctl_time_pre) * self.weeder_speed > self.param['weeder']['cmd_dist']:

            # fill the weeder_cmd_buff
            step_num = int(np.floor((cur_time - self.ctl_time_pre) * self.weeder_speed / self.param['weeder']['cmd_dist']))
            for step in range(step_num):
                self.weeder_cmd_buff[0:self.weeder_cmd_buff.shape[0] - 1] = self.weeder_cmd_buff[1:self.weeder_cmd_buff.shape[0]].copy()
                self.weeder_cmd_buff[self.weeder_cmd_buff.shape[0] - 1] = weeder_cmd

            # use delayed weeder cmd or the newest weeder cmd
            if self.weeder_cmd_delay:
                self.weeder_cmd = self.weeder_cmd_buff[0]
            else:
                # self.weeder_cmd = self.weeder_cmd_buff[-1]  # extract the most recent position shift
                self.weeder_cmd = weeder_cmd

            self.ctl_time_pre = cur_time

            # send the control cmd, i.e. the absolute position shift of the weeder
            msg = Float32MultiArray()
            msg.data = [self.weeder_cmd]
            self.weeder_control_pub.publish(msg)
            if self.verbose:
                rospy.loginfo('sent weeder_cmd = %f' % (self.weeder_cmd))
            return self.weeder_cmd
        else:
            return None

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
    rospy.init_node('lane_det_node', anonymous=True)
    lane_detector_obj = lane_detector()

    # if camera selection is failed, return
    if lane_detector_obj.cam_to_use==-1:
        rospy.logwarn('lane_det: camera selection failed, ending ...')
        return False

    # the main loop for line detection
    while not rospy.is_shutdown():
        lane_detector_obj.img_cb()


# the main entrance
if __name__ == '__main__':
    main(sys.argv)

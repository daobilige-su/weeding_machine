#! /usr/bin/env python
import numpy as np
import rospy
import sys
import tf
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float32
import ros_numpy
from transform_tools import *

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch

import rospkg
import yaml

class image_processor:
    def __init__(self):
        # locate ros pkg
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'

        # load key params
        params_filename = self.pkg_path + 'cfg/' + 'param.yaml'
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)
        self.img_size = np.array([self.param['cam']['height'], self.param['cam']['width']])
        self.bird_pixel_size = 0.01
        self.max_line_width = 0.5
        self.verbose = 0

        # yolo5 detection
        self.yolo5_model_path = self.pkg_path + 'weights/best.pt'
        self.model = torch.hub.load(self.pkg_path + 'yolov5', 'custom', path=self.yolo5_model_path, source='local')
        self.det_x_thr = 20

        # cam vars, pubs and subs
        self.left_img = None
        self.right_img = None
        self.cam_to_use = 0
        self.bridge = CvBridge()
        self.left_img_sub = rospy.Subscriber("/left_rgb_cam/image_raw", Image, self.left_img_cb)
        self.right_img_sub = rospy.Subscriber("/right_rgb_cam/image_raw", Image, self.right_img_cb)
        self.seg_img_pub = rospy.Publisher('/seg_img/image_raw', Image, queue_size=2)
        self.line_img_pub = rospy.Publisher('/line_img/image_raw', Image, queue_size=2)

        # weeder pub, sub and vars
        self.weeder_speed = 1
        self.weeder_control_pub = rospy.Publisher('/weeder_cmd', Float32MultiArray, queue_size=2)
        self.weeder_speed_sub = rospy.Subscriber("/weeder_speed", Float32, self.weeder_speed_cb)
        self.mid_line_y = None
        self.control_bias = 0
        self.weeder_y = 0
        self.mid_y_buff = np.zeros((35,))
        self.ctl_time_pre = -1
        self.ctl_cmd = 0

        # get camera intrinsic
        cam_info = rospy.wait_for_message("/left_rgb_cam/camera_info", CameraInfo)
        self.cam_K = np.array(cam_info.K).reshape((3, 3))
        # get camera extrinsic
        self.cam_T = transform_trans_ypr_to_matrix(np.array([[0, 0, -0.65, 0, 0, np.pi/4.0]]))
        # camera homography
        self.bird_img_size = None
        self.cam_H = None
        self.compute_homography()

        return

    # sub for weeder weeder_speed, simply store it in member var
    def weeder_speed_cb(self, msg):
        self.weeder_speed = msg.data

    # sub for left cam img, store it in member var and call line detection if left cam is selected.
    def left_img_cb(self, msg):
        # store var
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        # call line detection if selected
        if self.cam_to_use == 0:
            self.img_cb(self.left_img)

    # sub for right cam img, store it in member var and call line detection if right cam is selected.
    def right_img_cb(self, msg):
        # store var
        try:
            self.right_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        # call line detection if selected
        if self.cam_to_use == 1:
            self.img_cb(self.right_img)

    # the main loop: detect line and send back the result
    def img_cb(self, cv_image):

        # (1) segment plants in the image
        # segment plants with yolov5
        plant_seg = self.segment_plants_yolo5(cv_image)

        # publish segmentation results
        plant_seg_pub = np.asarray((plant_seg * 255).astype(np.uint8))
        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_pub, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        self.seg_img_pub.publish(ros_image)

        # (2) detect lanes
        # extract lines with the two step hough transform method
        lines, plant_seg_lines = self.detect_lanes(plant_seg)
        if lines is None:
            rospy.logwarn('no lines are detected, skipping this image and returning ...')
            return

        # publish line detection results
        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_lines, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.line_img_pub.publish(ros_image)

        # (3) control weeder
        # compute and send weeder cmd
        self.control_weeder(lines)

        # return once everything successes
        return

    def segment_plants_yolo5(self, np_image):
        image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

        # detection
        results = self.model(image_rgb)

        # image_seg = image_rgb.copy()
        image_seg_bn = np.zeros((image_rgb.shape[0],image_rgb.shape[1]))
        # mpl.use('TkAgg')
        # plt.imshow(image_seg, cmap='gray')
        # plt.show()

        # bbox coords, only select id=0, i.e. plants
        bboxes = results.xyxy[0].cpu().numpy()
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            if int(cls) == 0:  # only select id=0, i.e. plants
                if y1>self.det_x_thr and y2>self.det_x_thr:
                    # image_seg = cv2.rectangle(image_seg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
                    image_seg_bn = cv2.rectangle(image_seg_bn, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)

        # mpl.use('TkAgg')
        # plt.imshow(image_seg, cmap='gray')
        # plt.show()
        #
        # mpl.use('TkAgg')
        # plt.imshow(image_seg_bn, cmap='gray')
        # plt.show()

        return image_seg_bn

    def detect_lanes(self, plant_seg):

        detect_roi = [0.0, 1.0, 0.2, 1.0] # u_min, u_max, v_min, v_max, in percentage
        plant_seg_roi = plant_seg.copy()
        plant_seg_roi[:, :int(detect_roi[0]*plant_seg_roi.shape[1])] = 0
        plant_seg_roi[:, int(detect_roi[1]*plant_seg_roi.shape[1]):] = 0
        plant_seg_roi[:int(detect_roi[2] * plant_seg_roi.shape[0]), :] = 0
        plant_seg_roi[int(detect_roi[3] * plant_seg_roi.shape[0]):, :] = 0
        plt.imshow(plant_seg_roi, cmap='gray')

        plant_seg_rgb = np.asarray((plant_seg_roi * 255).astype(np.uint8))
        # plant_seg_cv = cv2.

        bird_eye_view = cv2.warpPerspective(plant_seg_rgb, self.H, (self.bird_img_size[0], self.bird_img_size[1]), flags=cv2.INTER_NEAREST)

        # plt.imshow(bird_eye_view, cmap='gray')

        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), 10, None, 0, 0, np.deg2rad(-20), np.deg2rad(20))
        if lines is None:
            rospy.logwarn('HoughLines: no lines are detected.')
            return None, None
        lines2d = lines.reshape((lines.shape[0], lines.shape[2]))
        lines2d[lines2d[:, 0]<0, 1] = ((lines2d[lines2d[:, 0]<0, 1]+np.pi) + np.pi) % (2*np.pi) - np.pi # wrap to pi
        lines2d[lines2d[:, 0]<0, 0] = -lines2d[lines2d[:, 0]<0, 0]
        lines_theta = lines2d[0:10, 1]
        if self.seg_type==2:
            lines_r = lines2d[0:10, 0]
            thetas = np.zeros((10,))
            idx = 0
            for i in range(lines_theta.shape[0]):
                if bird_eye_view.shape[1]*0.4 < lines_r[i] < bird_eye_view.shape[1]*0.6:
                    thetas[idx] = lines_theta[i]
                    idx = idx+1
            lines_theta = thetas[0:idx]
            if idx == 0:
                rospy.logwarn('detect_lanes: not enough middle lines detected.')
                return None, None

        theta = np.mean(lines_theta)

        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), 10, None, 0, 0, theta, theta+np.deg2rad(1))
        if lines is None:
            rospy.logwarn('HoughLines: no lines are detected.')
            return None, None
        lines_r = lines.reshape((lines.shape[0], lines.shape[2]))[:, 0]

        r_thr = self.max_line_width*0.5/self.bird_pixel_size # combine lines close to half lane width
        lines_r_asc = np.sort(lines_r)

        lines_r_sel = np.zeros((2, lines_r_asc.shape[0]))
        lines_r_sel[1, :] = theta

        idx = 0
        n = 0
        for i in range(lines_r_asc.shape[0]):
            r = lines_r_asc[i]
            if i==0:
                lines_r_sel[0, idx] = r
                n = 1
            else:
                if abs(r-lines_r_asc[i-1])<r_thr:
                    lines_r_sel[0, idx] = lines_r_sel[0, idx]+r
                    n = n+1
                else:
                    lines_r_sel[0, idx] = lines_r_sel[0, idx]/n

                    idx = idx+1
                    lines_r_sel[0, idx] = lines_r_sel[0, idx]+r
                    n=1
        lines_r_sel[0, idx] = lines_r_sel[0, idx] / n
        lines_r_sel = lines_r_sel[:, 0:idx+1]

        bird_eye_view_lines = np.stack((bird_eye_view,)*3, axis=-1)

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

                pt_edge_x = (rho-bird_eye_view.shape[0]*b)/a
                lines_x_theta[0, i] = pt_edge_x
                cv2.circle(bird_eye_view_lines, (int(bird_eye_view.shape[1]/2.0), bird_eye_view.shape[0]), 5, (255, 0, 0), 1)
                cv2.circle(bird_eye_view_lines, (int(pt_edge_x), bird_eye_view.shape[0]), 5, (0, 0, 255), 1)

        # plt.imshow(bird_eye_view_lines, cmap='gray')

        return lines_x_theta, bird_eye_view_lines

    def control_weeder(self, lines):
        lines_u_pix = lines[0, :]+self.control_bias
        lines_y = -(lines_u_pix-(self.bird_img_size[0]/2.0))*self.bird_pixel_size

        # if self.mid_line_y is None:
        #     idx = np.argmin(abs(lines_y - 0))
        #     self.mid_line_y = lines_y[idx]
        # else:
        #     idx = np.argmin(abs(lines_y - self.mid_line_y))
        #     self.mid_line_y = lines_y[idx]
        if (lines_y[lines_y > 0].shape[0]==0) or (lines_y[lines_y < 0].shape[0]==0):
            rospy.logwarn('no lane at left and right side is detected.')
            return

        pos_idx = np.argmin(abs(lines_y[lines_y > 0] - 0))
        pos_y = (lines_y[lines_y > 0])[pos_idx]
        neg_idx = np.argmin(abs(lines_y[lines_y < 0] - 0))
        neg_y = (lines_y[lines_y < 0])[neg_idx]

        mid_y = 0
        if pos_y<self.max_line_width and neg_y>-self.max_line_width:
            mid_y = (pos_y+neg_y)/2.0
        else:
            rospy.logwarn('lane y exceed the max lane width')
            return

        speed = self.weeder_speed * 0.5
        cur_time = rospy.get_time()
        if (cur_time-self.ctl_time_pre)*speed>0.1:
            self.mid_y_buff[0:self.mid_y_buff.shape[0]-1] = self.mid_y_buff[1:self.mid_y_buff.shape[0]].copy()
            self.mid_y_buff[self.mid_y_buff.shape[0]-1] = mid_y
            if self.verbose:
                print(self.mid_y_buff)
            self.ctl_time_pre = cur_time

            # ctl_cmd = 0
            # if self.mid_y_buff[0] > 0.10*self.max_line_width:
            #     ctl_cmd = -0.05
            # elif self.mid_y_buff[0] < -0.25*self.max_line_width:
            #     ctl_cmd = 0.05

            # self.weeder_y = self.weeder_y+ctl_cmd
            self.weeder_y = self.mid_y_buff[-1]

            u = self.weeder_y*0.5  # p controller
            if abs(u)<0.01:
                u = np.sign(u)*0.01

            if self.weeder_y<0:
                self.ctl_cmd = self.ctl_cmd + u
            else:
                self.ctl_cmd = self.ctl_cmd + u

            msg = Float32MultiArray()
            msg.data = [self.ctl_cmd]
            self.weeder_control_pub.publish(msg)
            print('ctl_cmd = %f \n' % (self.ctl_cmd))
        else:
            return

        pass

    def compute_homography(self):
        uv = np.array([[0, self.img_size[1]-1, self.img_size[1]-1, 0], [0, 0, self.img_size[0]-1, self.img_size[0]-1]])

        K = self.cam_K

        # Rt_mc = transform_trans_ypr_to_matrix(np.array([0, 0, 1.2, np.deg2rad(-90), 0, np.deg2rad(-90 - 45)]))
        Rt_mc = self.cam_T
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

        XY = np.divide(np.array([[k22 * k34 - k24 * k32, k14 * k32 - k12 * k34, k12 * k24 - k14 * k22],
                                 [k24 * k31 - k21 * k34, k11 * k34 - k14 * k31, k14 * k21 - k11 * k24]]) \
                       @ np.block([[uv], [np.ones((1, uv.shape[1]))]]), \
                       np.array([[k21 * k32 - k22 * k31, k12 * k31 - k11 * k32, k11 * k22 - k12 * k21],
                                 [k21 * k32 - k22 * k31, k12 * k31 - k11 * k32, k11 * k22 - k12 * k21]]) \
                       @ np.block([[uv], [np.ones((1, uv.shape[1]))]]))

        outputpts = np.float32([[XY[0, 0], XY[1, 0]], [XY[0, 1], XY[1, 1]], [XY[0, 2], XY[1, 2]], [XY[0, 3], XY[1, 3]]])
        inputpts = np.float32([[uv[0, 0], uv[1, 0]], [uv[0, 1], uv[1, 1]], [uv[0, 2], uv[1, 2]], [uv[0, 3], uv[1, 3]]])

        outputpts[:, 1] = outputpts[:, 1] - np.min(outputpts[:, 1])
        outputpts[:, 0] = outputpts[:, 0] - np.min(outputpts[:, 0])

        outputpts = np.round(outputpts/self.bird_pixel_size)
        self.bird_img_size = np.max(outputpts, axis=0)

        self.H = cv2.getPerspectiveTransform(inputpts, outputpts)


def main(args):
    rospy.init_node('image_processor_node', anonymous=True)
    img_proc = image_processor()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

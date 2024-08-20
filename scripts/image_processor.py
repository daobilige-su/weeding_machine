#! /usr/bin/env python
import numpy as np
import rospy
import sys
import tf
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import ros_numpy
from transform_tools import *

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib as mpl


import rospkg

class image_processor:
    def __init__(self):
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'
        self.seg_type = 1

        # yolo5 detection
        self.det_x_thr = None
        self.yolo5_model_path = None
        self.model = None
        if self.seg_type == 2:
            import torch
            self.yolo5_model_path = self.pkg_path + 'weights/best.pt'
            self.model = torch.hub.load(self.pkg_path + 'yolov5', 'custom', path=self.yolo5_model_path, source='local')
            self.det_x_thr = 20

        self.tf_listener = tf.TransformListener()

        self.bridge = CvBridge()
        self.pc2_sub = rospy.Subscriber("/left_rgb_cam/image_raw", Image, self.img_cb)
        self.seg_img_pub = rospy.Publisher('/seg_img/image_raw', Image, queue_size=2)
        self.line_img_pub = rospy.Publisher('/line_img/image_raw', Image, queue_size=2)
        self.weeder_control_pub = rospy.Publisher('/weeder_cmd', Float32MultiArray, queue_size=2)

        # mid line track
        self.mid_line_y = None
        self.control_bias = 0
        self.max_line_width = 1.0
        self.weeder_y = 0
        self.mid_y_buff = np.zeros((35,))
        self.ctl_time_pre = -1
        self.ctl_cmd = 0

        # get camera intrinsic
        cam_info = rospy.wait_for_message("/left_rgb_cam/camera_info", CameraInfo)
        self.cam_K = np.array(cam_info.K).reshape((3, 3))

        # get camera extrinsic
        self.cam_T = None
        for i in range(10):
            try:
                (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/left_rgb_cam', rospy.Time(0))
                # M = transform_trans_quat_to_matrix(np.array([[trans[0]], [trans[1]], [trans[2]], [rot[0]], [rot[1]], [rot[2]], [rot[3]]]))
                M = transform_trans_ypr_to_matrix(np.array([[0, 0, 0, 0, 0, -np.pi/4.0]]))
                self.cam_T = M[0:3, :]
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn('image_processor: tf lookup failed')
                if i == 9:
                    return
                else:
                    rospy.sleep(1.0)

        # camera homography
        self.cam_H = self.cam_K @ self.cam_T

        pass

    def img_cb(self, msg):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # (1) segment plants in the image
        if self.seg_type == 1:
            np_image = np.asarray(cv_image, dtype=np.float64)  # BRG
            plant_seg = self.segment_plants_exg(np_image)
        elif self.seg_type == 2:
            plant_seg = self.segment_plants_yolo5(cv_image)
        else:
            rospy.logwarn('unknown plant segmentation type. skipping')
            return

        plant_seg_pub = np.asarray((plant_seg * 255).astype(np.uint8))

        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_pub, "mono8")
        except CvBridgeError as e:
            print(e)
        self.seg_img_pub.publish(ros_image)

        # (2) detect lanes
        lines, plant_seg_lines = self.detect_lanes(plant_seg)
        if lines is None:
            return

        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_lines, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.line_img_pub.publish(ros_image)

        # (3) control weeder
        self.control_weeder(lines)


        pass

    def segment_plants_exg(self, np_image):
        # mpl.__version__

        # compute ExG index
        exg_index = 2 * np_image[:, :, 1] - (np_image[:, :, 2] + np_image[:, :, 0])  # ExG = 2G-(R+B)
        # debug
        # plt.imshow(exg_index, cmap='gray')

        # Otsu thresholding
        # otsu_thr = filters.threshold_otsu(exg_index)
        otsu_thr = 80
        plant_seg = (exg_index > otsu_thr).astype(float)
        # debug
        # plt.imshow(plant_seg, cmap='gray')

        return plant_seg

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

        detect_roi = [0.0, 1.0, 0.2, 1.0] # u_min, u_max, v_min, v_max
        plant_seg_roi = plant_seg.copy()
        plant_seg_roi[:, :int(detect_roi[0]*plant_seg_roi.shape[1])] = 0
        plant_seg_roi[:, int(detect_roi[1]*plant_seg_roi.shape[1]):] = 0
        plant_seg_roi[:int(detect_roi[2] * plant_seg_roi.shape[0]), :] = 0
        plant_seg_roi[int(detect_roi[3] * plant_seg_roi.shape[0]):, :] = 0
        plt.imshow(plant_seg_roi, cmap='gray')

        plant_seg_rgb = np.asarray((plant_seg_roi * 255).astype(np.uint8))
        # plant_seg_cv = cv2.

        outputpts = np.float32([[0, 0], [280-1, 0], [160-1, 180-1], [120-1, 180-1]])
        inputpts = np.float32([[0, 0], [320 - 1, 0], [320 - 1, 240 - 1], [0, 240 - 1]])

        m = cv2.getPerspectiveTransform(inputpts, outputpts)
        bird_eye_view = cv2.warpPerspective(plant_seg_rgb, m, (280, 180), flags=cv2.INTER_NEAREST)

        # plt.imshow(bird_eye_view, cmap='gray')


        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), 10, None, 0, 0, np.deg2rad(-20), np.deg2rad(20))
        if lines is None:
            rospy.logwarn('HoughLines: no lane_det_lines are detected.')
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
                rospy.logwarn('detect_lanes: not enough middle lane_det_lines detected.')
                return None, None

        theta = np.mean(lines_theta)

        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), 10, None, 0, 0, theta, theta+np.deg2rad(1))
        if lines is None:
            rospy.logwarn('HoughLines: no lane_det_lines are detected.')
            return None, None
        lines_r = lines.reshape((lines.shape[0], lines.shape[2]))[:, 0]

        r_thr = 5
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
        lines_r = lines[0, :]+self.control_bias
        lines_y = -(lines_r-(280.0/2.0))*(0.5/20.0)

        # if self.mid_line_y is None:
        #     idx = np.argmin(abs(lines_y - 0))
        #     self.mid_line_y = lines_y[idx]
        # else:
        #     idx = np.argmin(abs(lines_y - self.mid_line_y))
        #     self.mid_line_y = lines_y[idx]
        pos_idx = np.argmin(abs(lines_y[lines_y > 0] - 0))
        pos_y = (lines_y[lines_y > 0])[pos_idx]
        neg_idx = np.argmin(abs(lines_y[lines_y < 0] - 0))
        neg_y = (lines_y[lines_y < 0])[neg_idx]

        mid_y = 0
        if pos_y<self.max_line_width and neg_y>-self.max_line_width:
            mid_y = (pos_y+neg_y)/2.0

        speed = 0.2*0.8
        cur_time = rospy.get_time()
        if (cur_time-self.ctl_time_pre)*speed>0.1:
            self.mid_y_buff[0:self.mid_y_buff.shape[0]-1] = self.mid_y_buff[1:self.mid_y_buff.shape[0]].copy()
            self.mid_y_buff[self.mid_y_buff.shape[0]-1] = mid_y
            print(self.mid_y_buff)
            self.ctl_time_pre = cur_time

            # ctl_cmd = 0
            # if self.mid_y_buff[0] > 0.10*self.max_lane_width:
            #     ctl_cmd = -0.05
            # elif self.mid_y_buff[0] < -0.25*self.max_lane_width:
            #     ctl_cmd = 0.05

            # self.weeder_y = self.weeder_y+ctl_cmd
            self.weeder_y = self.mid_y_buff[-1]

            if self.weeder_y<0:
                self.ctl_cmd = self.ctl_cmd-0.01
            else:
                self.ctl_cmd = self.ctl_cmd + 0.01

            msg = Float32MultiArray()
            msg.data = [self.ctl_cmd]
            self.weeder_control_pub.publish(msg)
        else:
            return

        pass


def main(args):
    rospy.init_node('image_processor_node', anonymous=True)
    img_proc = image_processor()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

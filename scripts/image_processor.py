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
from skimage import filters


class image_processor:
    def __init__(self):
        self.tf_listener = tf.TransformListener()

        self.bridge = CvBridge()
        self.pc2_sub = rospy.Subscriber("/front_rgb_cam/image_raw", Image, self.img_cb)
        self.seg_img_pub = rospy.Publisher('/seg_img/image_raw', Image, queue_size=2)
        self.line_img_pub = rospy.Publisher('/line_img/image_raw', Image, queue_size=2)
        self.weeder_control_pub = rospy.Publisher('/weeder_cmd', Float32MultiArray, queue_size=2)

        # mid line track
        self.mid_line_y = None

        # get camera intrinsic
        cam_info = rospy.wait_for_message("/front_rgb_cam/camera_info", CameraInfo)
        self.cam_K = np.array(cam_info.K).reshape((3, 3))

        # get camera extrinsic
        self.cam_T = None
        for i in range(10):
            try:
                (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/front_rgb_cam', rospy.Time(0))
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

        # cv2.imshow('cv_image_wood_show', cv_image)
        # cv2.waitKey(0)

        np_image = np.asarray(cv_image, dtype=np.float64) # BRG

        # segment plants in the image
        plant_seg = self.segment_plants(np_image)
        plant_seg_pub = np.asarray((plant_seg * 255).astype(np.uint8))

        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_pub, "mono8")
        except CvBridgeError as e:
            print(e)
        self.seg_img_pub.publish(ros_image)

        # detect lanes
        lines, plant_seg_lines = self.detect_lanes(plant_seg)

        try:
            ros_image = self.bridge.cv2_to_imgmsg(plant_seg_lines, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.line_img_pub.publish(ros_image)

        # control weeder
        self.control_weeder(lines)


        pass

    def detect_lanes(self, plant_seg):

        plant_seg = np.asarray((plant_seg*255).astype(np.uint8))
        # plant_seg_cv = cv2.

        outputpts = np.float32([[0, 0], [280-1, 0], [160-1, 180-1], [120-1, 180-1]])
        inputpts = np.float32([[0, 0], [320 - 1, 0], [320 - 1, 240 - 1], [0, 240 - 1]])

        m = cv2.getPerspectiveTransform(inputpts, outputpts)
        bird_eye_view = cv2.warpPerspective(plant_seg, m, (280, 180), flags=cv2.INTER_NEAREST)

        # plt.imshow(bird_eye_view, cmap='gray')


        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), 50, None, 0, 0)
        lines2d = lines.reshape((lines.shape[0], lines.shape[2]))
        lines2d[lines2d[:, 0]<0, 1] = ((lines2d[lines2d[:, 0]<0, 1]+np.pi) + np.pi) % (2*np.pi) - np.pi # wrap to pi
        lines2d[lines2d[:, 0]<0, 0] = -lines2d[lines2d[:, 0]<0, 0]
        lines_theta = lines2d[0:10, 1]
        theta = np.mean(lines_theta)

        lines = cv2.HoughLines(bird_eye_view, 1, np.deg2rad(1), 10, None, 0, 0, theta, theta+np.deg2rad(1))
        lines_r = lines.reshape((lines.shape[0], lines.shape[2]))[:, 0]

        r_thr = 10
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

        if lines_r_sel is not None:
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

        # plt.imshow(bird_eye_view_lines, cmap='gray')

        return lines_r_sel, bird_eye_view_lines

    def segment_plants(self, np_image):
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

    def control_weeder(self, lines):
        lines_r = lines[0, :]
        lines_y = -(lines_r-(280.0/2.0))*(1.0/20.0)

        if self.mid_line_y is None:
            idx = np.argmin(abs(lines_y - 0))
            self.mid_line_y = lines_y[idx]
        else:
            idx = np.argmin(abs(lines_y - self.mid_line_y))
            self.mid_line_y = lines_y[idx]

        msg = Float32MultiArray()
        msg.data = [self.mid_line_y]
        self.weeder_control_pub.publish(msg)

        pass


def main(args):
    rospy.init_node('image_processor_node', anonymous=True)
    img_proc = image_processor()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

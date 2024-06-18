#! /usr/bin/env python
import numpy as np
import rospy
import sys
import tf
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import ros_numpy
from transform_tools import *
from visualization_msgs.msg import MarkerArray

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

from scipy import stats
from scipy import signal

from plot_marker import *


class line_detector:
    def __init__(self):
        self.tf_listener = tf.TransformListener()

        # get camera intrinsic
        cam_info = rospy.wait_for_message("/front_rgb_cam/camera_info", CameraInfo)
        self.cam_K = np.array(cam_info.K).reshape((3, 3))

        # get camera extrinsic
        self.cam_T = None
        for i in range(10):
            try:
                (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/front_rgb_cam', rospy.Time(0))
                M = transform_trans_quat_to_matrix(
                    np.array([[trans[0]], [trans[1]], [trans[2]], [rot[0]], [rot[1]], [rot[2]], [rot[3]]]))
                # M = transform_trans_ypr_to_matrix(np.array([[0, 0, 0, 0, 0, -np.pi/4.0]]))
                self.cam_T = M
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn('image_processor: tf lookup failed')
                if i == 9:
                    return
                else:
                    rospy.sleep(1.0)

        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber("/front_rgb_cam/image_raw", Image, self.img_cb)
        self.seg_img_pub = rospy.Publisher('/seg_img/image_raw', Image, queue_size=2)
        self.pc2_pub = rospy.Publisher('/seg_pc', PointCloud2, queue_size=5)

        self.line_img_pub = rospy.Publisher('/line_img/image_raw', Image, queue_size=2)
        self.line_marker_pub = rospy.Publisher('/line_marker', MarkerArray, queue_size=2)

        # mid line track
        self.mid_line_y = None

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

        # plant_seg to pc
        pc = self.img_to_pc(plant_seg)

        pc2_msg = self.numpy_to_pc2_msg(pc, msg.header)
        self.pc2_pub.publish(pc2_msg)

        # pc to lines
        lines = self.extract_lines(pc)

        mk_msg = self.show_lines(lines, msg.header)
        self.line_marker_pub.publish(mk_msg)

        # try:
        #     ros_image = self.bridge.cv2_to_imgmsg(plant_seg_lines, "bgr8")
        # except CvBridgeError as e:
        #     print(e)
        # self.line_img_pub.publish(ros_image)


        pass

    def extract_lines(self, pc, theta_reso=1.0, theta_pre=None, theta_range=None):

        ct = np.array([np.mean(pc[0, :]), np.mean(pc[1, :])])
        pc_ct = np.block([[pc[0, :] - ct[0]], [pc[1, :] - ct[1]], [pc[2, :]]])

        pc_ct_Y_range = np.array([np.min(pc_ct[1, :]), np.max(pc_ct[1, :])])

        if theta_pre is None:
            theta_pre = 0
            theta_range = np.array([-45.0, 45.0])
        else:
            if theta_range is None:
                rospy.logwarn('line_detect: theta_pre is provide, but theta_range is None.')
                return None

        hist_std = np.arange(theta_pre+theta_range[0], theta_pre+theta_range[1]+theta_reso, theta_reso)*0.0
        idx = 0
        for theta_deg in np.arange(theta_pre+theta_range[0], theta_pre+theta_range[1]+theta_reso, theta_reso):
            theta = np.deg2rad(theta_deg)
            R = ypr_to_matrix(np.array([theta, 0, 0]))
            pc_theta = R @ pc_ct

            Y = pc_theta[1, :]

            hist = np.histogram(Y, np.arange(pc_ct_Y_range[0], pc_ct_Y_range[1]+0.05, 0.05), density=True)
            # plt.hist(Y, np.arange(pc_ct_Y_range[0], pc_ct_Y_range[1]+0.05, 0.05), density=True)

            std = np.std(hist[0])
            hist_std[idx] = std

            idx = idx+1

        # plt.plot(hist_std)

        theta_opt = np.deg2rad(np.arange(theta_pre+theta_range[0], theta_pre+theta_range[1]+theta_reso, theta_reso)[np.argmax(hist_std)])
        R = ypr_to_matrix(np.array([theta_opt, 0, 0]))
        pc_theta = R @ pc
        Y = pc_theta[1, :]
        # hist = np.histogram(Y, np.arange(pc_ct_Y_range[0], pc_ct_Y_range[1] + 0.05, 0.05), density=True)
        Y_kde = stats.gaussian_kde(Y, bw_method=None)

        pc_theta_Y_range = np.array([np.min(Y), np.max(Y)])
        Y_kde_sam_idx = np.arange(pc_theta_Y_range[0], pc_theta_Y_range[1], 0.01)
        Y_kde_sam = Y_kde(Y_kde_sam_idx)

        plt.plot(Y_kde_sam_idx, Y_kde_sam)

        min_dist = 0.3
        min_samples = min_dist/0.01
        peaks = signal.find_peaks(Y_kde_sam, distance=min_samples)
        peaks_idx = peaks[0]
        peaks_h = Y_kde_sam[peaks_idx]
        peaks_Y = Y_kde_sam_idx[peaks_idx]

        r_opt = peaks_Y

        plt.plot(peaks_Y, peaks_h, 'k^')

        thetas = np.ones((1, r_opt.shape[0]))*theta_opt*(-1.0)

        line_edges = np.block([[np.ones((1, r_opt.shape[0]))*0.0], [np.ones((1, r_opt.shape[0]))*10.0]])

        return np.block([[thetas], [r_opt], [line_edges]])




    def img_to_pc(self, plant_seg):

        # if self.cam_T is None:
        #     rospy.logwarn('line_detect: self.cam_T is not ready yet.')
        #     return None

        img_size = np.array([plant_seg.shape[0], plant_seg.shape[1]])

        u = np.tile(np.array([range(img_size[1])]), (img_size[0], 1))
        v = np.tile(np.array([range(img_size[0])]).T, (1, img_size[1]))

        u_flat = u.reshape((-1,))
        v_flat = v.reshape((-1,))

        plant_seg_flat = plant_seg.reshape((-1,))
        plant_seg_u = u_flat[plant_seg_flat > 0]
        plant_seg_v = v_flat[plant_seg_flat > 0]

        uv = np.block([[plant_seg_u], [plant_seg_v]])

        # K = np.array([159.99999300617776, 0.0, 160.0, 0.0, 159.99999300617776, 120.0, 0.0, 0.0, 1.0])
        # K = np.array([159.99999300617776, 0.0, 160.00, 0.0, 159.99999300617776, 120.00, 0.0, 0.0, 1.0]).reshape((3, 3))
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
        Z = np.zeros((1, uv.shape[1]))

        P = np.block([[XY], [Z]])

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(P[0, :], P[1, :], P[2, :])
        # plt.show()

        return P

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

    def numpy_to_pc2_msg(self, np_pc, header):

        pc_array = np.zeros(np_pc.shape[1], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])

        pc_array['x'] = np_pc[0, :]
        pc_array['y'] = np_pc[1, :]
        pc_array['z'] = np_pc[2, :]

        pc2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(pc_array, header.stamp, 'base_link')
        pc2_msg.header.seq = header.seq

        return pc2_msg

    def show_lines(self, lines, msg_header):

        line_num = lines.shape[1]
        line_pts = np.zeros((6, line_num))
        for i in range(line_num):
            pts = transform_trans_ypr_to_matrix(np.array([0, 0, 0, lines[0, i], 0, 0])) @ np.array([[lines[2, i], lines[3, i]], [lines[1, i], lines[1, i]], [0, 0], [1, 1]])
            line_pts[:, i] = np.block([[pts[0:3, 0]], [pts[0:3, 1]]]).reshape((-1,))

        marker_array = plot_farm_lines(line_pts, 1, frame_id='base_link')

        return marker_array

def main(args):
    rospy.init_node('line_detect_node', anonymous=True)
    ld = line_detector()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

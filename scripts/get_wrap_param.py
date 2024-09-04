#! /usr/bin/env python
import numpy as np
import rospy
import sys
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float32
import ros_numpy
from transform_tools import *
from rospy.exceptions import ROSException

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib as mpl

import rospkg
import yaml

class wrap_param_provider():
    def __init__(self):
        # KEY PARAMS

        # locate ros pkg
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'

        # load key params
        params_filename = rospy.get_param('param_file')
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)

        self.verbose = self.param['verbose']
        self.img_size = np.array([self.param['cam']['height'], self.param['cam']['width']])
        self.left_cam_topic = self.param['cam']['left']['topic']
        self.right_cam_topic = self.param['cam']['right']['topic']

        self.bridge = CvBridge()

    # get wrap params for both left and right camera image
    def param_set(self):
        mpl.use('TkAgg')

        # left cam wrap param
        userinput = input("estimate left cam wrap param? (y):")
        if (userinput == 'y') or (userinput == 'Y'):
            # (1) obtain camera images
            rospy.loginfo('waiting 100s for left camera image of topic [' + self.left_cam_topic + ']')
            try:
                img_msg = rospy.wait_for_message(self.left_cam_topic, Image, 100)
            except ROSException as e:
                rospy.logwarn('image not received after 100s.')
                rospy.logwarn(e)
                return False
            rospy.loginfo('camera image received. ')

            cv_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(img_msg, "bgr8"))

            fig, ax = plt.subplots()
            ax.imshow(cv_img)

            cid = fig.canvas.mpl_connect('button_press_event', self.mouse_onclick)

    def mouse_onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))




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
    rospy.init_node('get_wrap_param_node', anonymous=True)
    wpp = wrap_param_provider()

    # compute wrap param
    wpp.param_set()


if __name__ == '__main__':
    main(sys.argv)
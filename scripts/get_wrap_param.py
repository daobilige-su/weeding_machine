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
        self.wrap_img_size_u = self.param['lane_det']['wrap_img_size_u']

        self.bridge = CvBridge()
        self.click_pt_type = None
        self.click_pts = None

    # get wrap params for both left and right camera image
    def param_set(self):
        mpl.use('TkAgg')

        left_cam_wrap_param = None
        right_cam_wrap_param = None

        # left cam wrap param
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        userinput = input("estimate left cam wrap param? (y/n):")
        if (userinput == 'y') or (userinput == 'Y'):
            # obtain camera images
            print('waiting 100s for left camera image of topic [' + self.left_cam_topic + ']')
            try:
                img_msg = rospy.wait_for_message(self.left_cam_topic, Image, 100)
            except ROSException as e:
                rospy.logwarn('image not received after 100s.')
                rospy.logwarn(e)
                return False
            print('camera image received. ')
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
            print('start left camera wrap param estimation.')

            cv_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(img_msg, "bgr8"))

            left_cam_wrap_param = self.param_set_for_one_cam(cv_img)

            print('+++++++++++++++++++++++++++++++++++++++++++++++')
        else:
            print('left cam wrap param will not be estimated.')
            print('+++++++++++++++++++++++++++++++++++++++++++++++')

        # right cam wrap param
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        userinput = input("estimate right cam wrap param? (y/n):")
        if (userinput == 'y') or (userinput == 'Y'):
            # obtain camera images
            print('waiting 100s for right camera image of topic [' + self.right_cam_topic + ']')
            try:
                img_msg = rospy.wait_for_message(self.right_cam_topic, Image, 100)
            except ROSException as e:
                rospy.logwarn('image not received after 100s.')
                rospy.logwarn(e)
                return False
            print('camera image received. ')
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
            print('start right camera wrap param estimation.')

            cv_img = self.resize_img_keep_scale(self.bridge.imgmsg_to_cv2(img_msg, "bgr8"))

            right_cam_wrap_param = self.param_set_for_one_cam(cv_img)

            print('+++++++++++++++++++++++++++++++++++++++++++++++')
        else:
            print('right cam wrap param will not be estimated.')
            print('+++++++++++++++++++++++++++++++++++++++++++++++')

        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        print('In summary:')
        print('left camera wrap param is: ')
        if left_cam_wrap_param is not None:
            print(np.array2string(left_cam_wrap_param, separator=', '))
        else:
            print('[]')
        print('right camera wrap param is: ')
        if right_cam_wrap_param is not None:
            print(np.array2string(right_cam_wrap_param, separator=', '))
        else:
            print('[]')
        print('+++++++++++++++++++++++++++++++++++++++++++++++')

    def param_set_for_one_cam(self, cv_img):
        wrap_param = None

        fig, ax = plt.subplots()
        ax.imshow(cv_img)
        plt.pause(1)

        cid = fig.canvas.mpl_connect('button_press_event', self.mouse_onclick)

        cv_img_draw = cv_img.copy()
        cv_img_draw_fix = cv_img.copy()
        self.click_pts = np.zeros((2, 7))

        # (1) left line
        left_line_k_b = None
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        print('let us fix the left line.')
        for n in range(10):
            cv_img_draw = cv_img_draw_fix.copy()
            ax.imshow(cv_img_draw)
            plt.pause(1)

            self.click_pt_type = 1
            input('click on the image for left line point 1, press ENTER after the click.')
            plt.pause(0.1)
            cv2.circle(cv_img_draw, (int(self.click_pts[0, 0]), int(self.click_pts[1, 0])), 5, (255, 0, 0), 5)
            ax.imshow(cv_img_draw)
            plt.pause(1)

            self.click_pt_type = 2
            input('click on the image for left line point 2, press ENTER after the click.')
            plt.pause(0.1)
            cv2.circle(cv_img_draw, (int(self.click_pts[0, 1]), int(self.click_pts[1, 1])), 5, (255, 0, 0), 5)
            ax.imshow(cv_img_draw)
            plt.pause(1)

            # u = k*v+b
            k = np.tan((self.click_pts[0, 0]-self.click_pts[0, 1])/(self.click_pts[1, 0]-self.click_pts[1, 1]))
            b = self.click_pts[0, 0] - k * self.click_pts[1, 0]

            pt_up = np.array([b, 0])
            pt_down = np.array([k * (self.img_size[0] - 1)+ b, self.img_size[0]-1])

            cv2.line(cv_img_draw, (int(pt_up[0]), int(pt_up[1])), (int(pt_down[0]), int(pt_down[1])), (0, 255, 0), 1, cv2.LINE_AA)
            ax.imshow(cv_img_draw)
            plt.pause(1)

            userinput = input("confirm this? (y/n):")
            if (userinput == 'y') or (userinput == 'Y'):
                left_line_k_b = np.array([k, b])
                break
            else:
                print('ok, let us start again.')

        if left_line_k_b is None:
            print('unfortunately, left line is not confirmed after trying 10 times. returning.')
            return

        cv_img_draw_fix = cv_img_draw.copy()

        # (2) right line
        right_line_k_b = None
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        print('let us fix the right line.')
        for n in range(10):
            cv_img_draw = cv_img_draw_fix.copy()
            ax.imshow(cv_img_draw)
            plt.pause(1)

            self.click_pt_type = 3
            input('click on the image for right line point 1, press ENTER after the click.')
            plt.pause(0.1)
            cv2.circle(cv_img_draw, (int(self.click_pts[0, 2]), int(self.click_pts[1, 2])), 5, (255, 0, 0), 5)
            ax.imshow(cv_img_draw)
            plt.pause(1)

            self.click_pt_type = 4
            input('click on the image for right line point 2, press ENTER after the click.')
            plt.pause(0.1)
            cv2.circle(cv_img_draw, (int(self.click_pts[0, 3]), int(self.click_pts[1, 3])), 5, (255, 0, 0), 5)
            ax.imshow(cv_img_draw)
            plt.pause(1)

            # u = k*v+b
            k = np.tan(
                (self.click_pts[0, 2] - self.click_pts[0, 3]) / (self.click_pts[1, 2] - self.click_pts[1, 3]))
            b = self.click_pts[0, 2] - k * self.click_pts[1, 2]

            pt_up = np.array([b, 0])
            pt_down = np.array([k * (self.img_size[0] - 1) + b, self.img_size[0] - 1])

            cv2.line(cv_img_draw, (int(pt_up[0]), int(pt_up[1])), (int(pt_down[0]), int(pt_down[1])), (0, 255, 0), 1, cv2.LINE_AA)
            ax.imshow(cv_img_draw)
            plt.pause(1)

            userinput = input("confirm this? (y/n):")
            if (userinput == 'y') or (userinput == 'Y'):
                right_line_k_b = np.array([k, b])
                break
            else:
                print('ok, let us start again.')

        if right_line_k_b is None:
            print('unfortunately, right line is not confirmed after trying 10 times. returning.')
            return

        cv_img_draw_fix = cv_img_draw.copy()

        # (3) v range
        v_range = None
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        userinput = input("use the whole height of the image (v range in [%d, %d])? (y/n):" % (0, self.img_size[0] - 1))
        if (userinput == 'y') or (userinput == 'Y'):
            self.click_pts[:, 4] = np.array([0, 0])
            self.click_pts[:, 5] = np.array([0, self.img_size[0]-1])

            v_range = np.array([self.click_pts[1, 4], self.click_pts[1, 5]])

            cv2.line(cv_img_draw, (0, int(v_range[0])), (self.img_size[1]-1, int(v_range[0])), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(cv_img_draw, (0, int(v_range[1])), (self.img_size[1]-1, int(v_range[1])), (0, 0, 255), 1, cv2.LINE_AA)
            ax.imshow(cv_img_draw)
            plt.pause(1)
        else:
            print('let us fix the v range.')
            for n in range(10):
                cv_img_draw = cv_img_draw_fix.copy()
                ax.imshow(cv_img_draw)
                plt.pause(1)

                self.click_pt_type = 5
                input('click on the image for top limit of v, press ENTER after the click.')
                plt.pause(0.1)
                v_range = np.array([self.click_pts[1, 4], self.click_pts[1, 5]])
                cv2.line(cv_img_draw, (0, int(v_range[0])), (self.img_size[1]-1, int(v_range[0])), (0, 0, 255), 1, cv2.LINE_AA)
                ax.imshow(cv_img_draw)
                plt.pause(1)

                self.click_pt_type = 6
                input('click on the image for bottom limit of v, press ENTER after the click.')
                plt.pause(0.1)
                v_range = np.array([self.click_pts[1, 4], self.click_pts[1, 5]])
                cv2.line(cv_img_draw, (0, int(v_range[1])), (self.img_size[1]-1, int(v_range[1])), (0, 0, 255), 1, cv2.LINE_AA)
                ax.imshow(cv_img_draw)
                plt.pause(1)

                userinput = input("confirm this? (y/n):")
                if (userinput == 'y') or (userinput == 'Y'):
                    break
                else:
                    v_range = None
                    print('ok, let us start again.')

            if v_range is None:
                print('unfortunately, v range is not confirmed after trying 10 times. returning.')
                return

        cv_img_draw_fix = cv_img_draw.copy()

        # (4) compute four corners
        # u = k*v+b
        four_corners = np.zeros((2, 4))
        four_corners[1, 0] = v_range[1]  # bot left v
        four_corners[1, 1] = v_range[1]  # bot right v
        four_corners[1, 2] = v_range[0]  # top left v
        four_corners[1, 3] = v_range[0]  # top right v

        four_corners[0, 0] = left_line_k_b[0] * four_corners[1, 0] + left_line_k_b[1]  # bot left u
        four_corners[0, 1] = right_line_k_b[0] * four_corners[1, 1] + right_line_k_b[1]  # bot right u
        four_corners[0, 2] = left_line_k_b[0] * four_corners[1, 2] + left_line_k_b[1]  # top left u
        four_corners[0, 3] = right_line_k_b[0] * four_corners[1, 3] + right_line_k_b[1]  # top right u

        cv2.circle(cv_img_draw, (int(four_corners[0, 0]), int(four_corners[1, 0])), 3, (0, 0, 255), 3)
        cv2.circle(cv_img_draw, (int(four_corners[0, 1]), int(four_corners[1, 1])), 3, (0, 0, 255), 3)
        cv2.circle(cv_img_draw, (int(four_corners[0, 2]), int(four_corners[1, 2])), 3, (0, 0, 255), 3)
        cv2.circle(cv_img_draw, (int(four_corners[0, 3]), int(four_corners[1, 3])), 3, (0, 0, 255), 3)
        ax.imshow(cv_img_draw)
        plt.pause(1)

        cv_img_draw_fix = cv_img_draw.copy()

        # (5) weeder location
        weeder_loc = None
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        userinput = input("use the middle of the two lines as weeder location? (y/n):")
        if (userinput == 'y') or (userinput == 'Y'):
            self.click_pts[:, 6] = np.array([(four_corners[0, 0]+four_corners[0, 1])/2.0, v_range[1]])
            weeder_loc = np.array([self.click_pts[0, 6], v_range[1]])

            cv2.circle(cv_img_draw, (int(weeder_loc[0]), int(weeder_loc[1])), 3, (0, 0, 0), 3)
            ax.imshow(cv_img_draw)
            plt.pause(1)
        else:
            print('let us fix the weeder location.')
            for n in range(10):
                cv_img_draw = cv_img_draw_fix.copy()
                ax.imshow(cv_img_draw)
                plt.pause(1)

                self.click_pt_type = 7
                input('click on the image near bottom v range for the weeder location, press ENTER after the click.')
                plt.pause(0.1)
                weeder_loc = np.array([self.click_pts[0, 6], v_range[1]])

                cv2.circle(cv_img_draw, (int(weeder_loc[0]), int(weeder_loc[1])), 3, (0, 0, 0), 3)
                ax.imshow(cv_img_draw)
                plt.pause(1)

                userinput = input("confirm this? (y/n):")
                if (userinput == 'y') or (userinput == 'Y'):
                    break
                else:
                    weeder_loc = None
                    print('ok, let us start again.')

            if weeder_loc is None:
                print('unfortunately, weeder location is not confirmed after trying 10 times. returning.')
                return

        cv_img_draw_fix = cv_img_draw.copy()

        # (6) compute wrap param and H matrix
        # wrap_param:[bot left u, bot right u, top left u, top right u, bot v, top v, weeder's u at bot]
        wrap_param = np.array([int(four_corners[0, 0]), int(four_corners[0, 1]), int(four_corners[0, 2]), int(four_corners[0, 3]),
                               int(four_corners[1, 0]), int(four_corners[1, 2]), int(weeder_loc[0])]).astype(int)
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        print('wrap_param is: ')
        print(wrap_param)

        # wrap_img size
        wrap_img_size = np.array([int(wrap_param[4] - wrap_param[5] + 1), int(2 * (wrap_param[1] - wrap_param[0]))])
        # wrap_H computation
        h_shift = (wrap_param[0] + wrap_param[1]) / 2.0 - (wrap_param[2] + wrap_param[3]) / 2.0
        if h_shift > 0:
            inputpts = np.float32([[0, wrap_img_size[0]], [wrap_img_size[1], wrap_img_size[0]], [wrap_img_size[1] - int(h_shift), 0],[0, 0]])
            outputpts = np.float32([[0, wrap_img_size[0]], [wrap_img_size[1], wrap_img_size[0]], [wrap_img_size[1], 0],[int(h_shift), 0]])
        else:
            inputpts = np.float32([[0, wrap_img_size[0]], [wrap_img_size[1], wrap_img_size[0]], [wrap_img_size[1], 0],[int(abs(h_shift)), 0]])
            outputpts = np.float32([[0, wrap_img_size[0]], [wrap_img_size[1], wrap_img_size[0]], [wrap_img_size[1] - int(abs(h_shift)), 0],[0, 0]])
        wrap_H = cv2.getPerspectiveTransform(inputpts, outputpts)

        # (7) fill, crop, transform, and resize the image
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        print('filling, cropping and transforming the image by now ...')
        cv_img_wrap = cv_img.copy()
        # fill two sides of the image will black image with half of its width
        u_half_size = int(self.img_size[1] / 2.0)
        cv_img_wrap = np.concatenate((cv_img_wrap[:, 0:u_half_size, :] * 0, cv_img_wrap, cv_img_wrap[:, 0:u_half_size, :] * 0), axis=1)
        ax.imshow(cv_img_wrap)
        plt.pause(2)
        # crop image, 1st vertical then 2nd horizontal
        cv_img_wrap = cv_img_wrap[wrap_param[5]:(wrap_param[4] + 1), :, :] # vertical crop
        cv_img_wrap = cv_img_wrap[:, int(u_half_size + wrap_param[0] - 0.5 * (wrap_param[1] - wrap_param[0])):
                                     int(u_half_size + wrap_param[1] + 0.5 * (wrap_param[1] - wrap_param[0])), :]
        ax.imshow(cv_img_wrap)
        plt.pause(2)
        # transform
        cv_img_wrap = cv2.warpPerspective(cv_img_wrap, wrap_H, (cv_img_wrap.shape[1], cv_img_wrap.shape[0]))
        ax.imshow(cv_img_wrap)
        plt.pause(2)
        # resize
        cv_img_wrap = cv2.resize(cv_img_wrap, (int(self.wrap_img_size_u), int(cv_img_wrap.shape[0] * (self.wrap_img_size_u / cv_img_wrap.shape[1]))))
        ax.imshow(cv_img_wrap)
        plt.pause(2)
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        userinput = input("Does the transformed image look good? (y/n):")
        if (userinput == 'y') or (userinput == 'Y'):
            print('great, then your wrap_param is confirmed to be: ')
            print(wrap_param)
        else:
            wrap_param = None
            print('oh no, please re-run this code all over again.')

        print('+++++++++++++++++++++++++++++++++++++++++++++++')

        return wrap_param

    def mouse_onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        if self.click_pt_type == 1:
            self.click_pts[:, 0] = np.array([event.xdata, event.ydata])
        elif self.click_pt_type == 2:
            self.click_pts[:, 1] = np.array([event.xdata, event.ydata])
        elif self.click_pt_type == 3:
            self.click_pts[:, 2] = np.array([event.xdata, event.ydata])
        elif self.click_pt_type == 4:
            self.click_pts[:, 3] = np.array([event.xdata, event.ydata])
        elif self.click_pt_type == 5:
            self.click_pts[:, 4] = np.array([event.xdata, event.ydata])
        elif self.click_pt_type == 6:
            self.click_pts[:, 5] = np.array([event.xdata, event.ydata])
        elif self.click_pt_type == 7:
            self.click_pts[:, 6] = np.array([event.xdata, event.ydata])


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
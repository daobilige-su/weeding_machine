#! /usr/bin/env python

import rospy
import sys
import tf


class TFManager:
    def __init__(self):
        self.send_rate = rospy.Rate(10)  # send with 10 hz
        self.br = tf.TransformBroadcaster()

    def tf_broadcast(self, trans, quat, child_frame, parent_frame):
        self.br.sendTransform((trans[0], trans[1], trans[2]), (quat[0], quat[1], quat[2], quat[3]), rospy.Time.now(),
                              child_frame, parent_frame)

    def send_tf(self):
        self.tf_broadcast([0.000004, 2.250000, -0.000002], [0.000000, 0.000000, 0.000000, 1.000000],
                          'left_weeder', 'base_link')
        self.tf_broadcast([0.000002, -2.249999, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000],
                          'right_weeder', 'base_link')
        self.tf_broadcast([1.619993, -0.000001, 0.650002], [-0.653281, 0.653281, -0.270598, 0.270598],
                          'left_rgb_cam', 'left_weeder')
        self.tf_broadcast([1.620018, 0.000000, 0.650000], [-0.653282, 0.653282, -0.270598, 0.270598],
                          'right_rgb_cam', 'right_weeder')
        self.tf_broadcast([0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000],
                          'left_depth_cam', 'left_rgb_cam')
        self.tf_broadcast([0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000],
                          'right_depth_cam', 'right_rgb_cam')


def main(args):
    rospy.init_node('tf_manager_node', anonymous=True)
    tfm = TFManager()
    try:
        while not rospy.is_shutdown():
            tfm.send_tf()
            tfm.send_rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)

#! /usr/bin/env python
import numpy as np
import rospy
import sys
import tf
from sensor_msgs.msg import PointCloud2
import ros_numpy
from transform_tools import *


class cloud_transform:
    def __init__(self):
        self.tf_listener = tf.TransformListener()

        self.pc2_sub = rospy.Subscriber("/front_depth_cam/points", PointCloud2, self.pc2_cb)
        self.pc2_pub = rospy.Publisher('/front_depth_cam/pc2_ur5_base', PointCloud2, queue_size=5)

    def pc2_cb(self, msg):
        pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg).T

        '''
        lookupTransform(target_frame, source_frame, time) -> (position, quaternion)
            Parameters:	        
                target_frame – transformation target frame in tf, string
                source_frame – transformation source frame in tf, string
                time – time of the transformation, use rospy.Time() to indicate most recent common time.        
            Returns:	        
                position as a translation (x, y, z) and orientation as a quaternion (x, y, z, w)
            Raises:	        
                tf.ConnectivityException, tf.LookupException, or tf.ExtrapolationException
        
        Returns the transform from source_frame to target_frame at time. 
        Raises one of the exceptions if the transformation is not possible. 
        Note that a time of zero means latest common time, so:      
        t.lookupTransform("a", "b", rospy.Time())        
        is equivalent to:        
        t.lookupTransform("a", "b", t.getLatestCommonTime("a", "b"))
        '''
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/ur5_base', '/front_depth_cam', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn('pc2_transform: tf lookup failed')
            return

        M = transform_trans_quat_to_matrix(np.array([[trans[0]], [trans[1]], [trans[2]], [rot[0]], [rot[1]], [rot[2]], [rot[3]]]))
        pc_transformed_np = np.block([[np.eye(3), np.zeros((3, 1))]]) @ M @ np.block([[pc_np], [np.ones((1, pc_np.shape[1]))]])

        pc_array = np.zeros(pc_transformed_np.shape[1], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])

        pc_array['x'] = pc_transformed_np[0, :]
        pc_array['y'] = pc_transformed_np[1, :]
        pc_array['z'] = pc_transformed_np[2, :]

        pc2_transformed_msg = ros_numpy.point_cloud2.array_to_pointcloud2(pc_array, msg.header.stamp, 'ur5_base')
        pc2_transformed_msg.header.seq = msg.header.seq

        self.pc2_pub.publish(pc2_transformed_msg)


def main(args):
    rospy.init_node('pc2_transform_node', anonymous=True)
    tfm = cloud_transform()

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

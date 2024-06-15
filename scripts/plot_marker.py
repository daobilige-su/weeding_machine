#!/usr/bin/env python
# -*- coding: utf-8 -*-

from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from visualization_msgs.msg import Marker
from transform_tools import *


def plot_pts(pts, marker_pub, id, frame_id='map', scale_size=0.01): # pts: Nx3
    marker_array = MarkerArray()
    marker_array.markers = []

    # pts
    pt_marker = Marker()
    pt_marker.header.frame_id = frame_id
    pt_marker.ns = "path_plan_" + "pt"
    pt_marker.id = id
    pt_marker.type = Marker.CUBE_LIST
    pt_marker.action = Marker.ADD
    pose = Pose()
    pose.orientation.w = 1.0
    pt_marker.pose = pose
    # when list is used, color needs to be 1.0 not 255, such a bug!
    pt_marker.color.r, pt_marker.color.g, pt_marker.color.b = (1.0, 1.0, 0)
    pt_marker.color.a = 1.0
    pt_marker.scale.x, pt_marker.scale.y, pt_marker.scale.z = (scale_size, scale_size, scale_size)

    pt_marker.points = []
    # pt_marker.colors = []
    pts_num = pts.shape[0]
    for i in range(pts_num):
        pt = Point()
        pt.x = pts[i, 0]
        pt.y = pts[i, 1]
        pt.z = pts[i, 2]
        pt_marker.points.append(pt)

        # color = ColorRGBA()
        # color.r, color.g, color.b = (255, 255, 255)
        # color.a = 1.0
        # pt_marker.colors.append(color)

    marker_array.markers.append(pt_marker)

    # publish marker_array
    marker_pub.publish(marker_array)


def plot_traj(traj_pts, marker_pub, start_id, frame_id='map', scale_size=0.01):  # traj_pts: Nx3
    marker_array = MarkerArray()
    marker_array.markers = []

    # traj_pts
    pt_marker = Marker()
    pt_marker.header.frame_id = frame_id
    pt_marker.ns = "path_plan_" + "traj_pts"
    pt_marker.id = start_id
    pt_marker.type = Marker.CUBE_LIST
    pt_marker.action = Marker.ADD
    pose = Pose()
    pose.orientation.w = 1.0
    pt_marker.pose = pose
    # when list is used, color needs to be 1.0 not 255, such a bug!
    pt_marker.color.r, pt_marker.color.g, pt_marker.color.b = (1.0, 0.0, 0)
    pt_marker.color.a = 1.0
    pt_marker.scale.x, pt_marker.scale.y, pt_marker.scale.z = (scale_size*2, scale_size*2, scale_size*2)

    pt_marker.points = []
    # pt_marker.colors = []
    traj_pts_num = traj_pts.shape[0]
    for i in range(traj_pts_num):
        pt = Point()
        pt.x = traj_pts[i, 0]
        pt.y = traj_pts[i, 1]
        pt.z = traj_pts[i, 2]
        pt_marker.points.append(pt)

        # color = ColorRGBA()
        # color.r, color.g, color.b = (255, 255, 255)
        # color.a = 1.0
        # pt_marker.colors.append(color)

    marker_array.markers.append(pt_marker)

    # traj line
    line_marker = Marker()
    line_marker.header.frame_id = frame_id
    line_marker.ns = "path_plan_" + "traj_line"
    line_marker.id = start_id+1
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    pose = Pose()
    pose.orientation.w = 1
    line_marker.pose = pose
    line_marker.color.r, line_marker.color.g, line_marker.color.b = (1.0, 0.0, 0.0)
    line_marker.color.a = 1.0
    line_marker.scale.x, line_marker.scale.y, line_marker.scale.z = (scale_size, scale_size, scale_size)

    line_marker.points = []
    for i in range(traj_pts_num):
        pt = Point()
        pt.x = traj_pts[i, 0]
        pt.y = traj_pts[i, 1]
        pt.z = traj_pts[i, 2]
        line_marker.points.append(pt)

    marker_array.markers.append(line_marker)

    # publish marker_array
    marker_pub.publish(marker_array)


def plot_farm_lines(lines, id, frame_id='base_link', scale_size=0.01):  # traj_pts: Nx3
    marker_array = MarkerArray()
    marker_array.markers = []

    # traj_pts
    pt_marker = Marker()
    pt_marker.header.frame_id = frame_id
    pt_marker.ns = "line_detect_" + "line_pts"
    pt_marker.id = id
    pt_marker.type = Marker.CUBE_LIST
    pt_marker.action = Marker.ADD
    pose = Pose()
    pose.orientation.w = 1.0
    pt_marker.pose = pose
    # when list is used, color needs to be 1.0 not 255, such a bug!
    pt_marker.color.r, pt_marker.color.g, pt_marker.color.b = (1.0, 0.0, 0)
    pt_marker.color.a = 1.0
    pt_marker.scale.x, pt_marker.scale.y, pt_marker.scale.z = (scale_size*2, scale_size*2, scale_size*2)

    pt_marker.points = []
    # pt_marker.colors = []
    line_num = lines.shape[1]
    for i in range(line_num):
        pt = Point()
        pt.x = lines[0, i]
        pt.y = lines[1, i]
        pt.z = lines[2, i]
        pt_marker.points.append(pt)

        pt = Point()
        pt.x = lines[3, i]
        pt.y = lines[4, i]
        pt.z = lines[5, i]
        pt_marker.points.append(pt)

        # color = ColorRGBA()
        # color.r, color.g, color.b = (255, 255, 255)
        # color.a = 1.0
        # pt_marker.colors.append(color)

    marker_array.markers.append(pt_marker)

    # line
    line_marker = Marker()
    line_marker.header.frame_id = frame_id
    line_marker.ns = "line_detect_" + "line"
    line_marker.id = id+1
    line_marker.type = Marker.LINE_LIST  # draw a line between each pair of points, so 0-1, 2-3, 4-5, ...
    line_marker.action = Marker.ADD
    pose = Pose()
    pose.orientation.w = 1
    line_marker.pose = pose
    line_marker.color.r, line_marker.color.g, line_marker.color.b = (1.0, 0.0, 0.0)
    line_marker.color.a = 1.0
    line_marker.scale.x, line_marker.scale.y, line_marker.scale.z = (scale_size, scale_size, scale_size)

    line_marker.points = []
    for i in range(line_num):
        pt = Point()
        pt.x = lines[0, i]
        pt.y = lines[1, i]
        pt.z = lines[2, i]
        line_marker.points.append(pt)

        pt = Point()
        pt.x = lines[3, i]
        pt.y = lines[4, i]
        pt.z = lines[5, i]
        line_marker.points.append(pt)

    marker_array.markers.append(line_marker)

    return marker_array


def plot_arrows(poses, marker_pub, start_id, frame_id='map', scale_size=0.01):  # poses: Nx6
    marker_array = MarkerArray()
    marker_array.markers = []

    poses_num = poses.shape[0]

    for n in range(poses_num):
        # arrow
        arrow_marker = Marker()
        arrow_marker.header.frame_id = frame_id
        arrow_marker.ns = "path_plan_" + "arrow"
        arrow_marker.id = start_id+n
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD

        # pose
        pose_trans = poses[n, 0:3].reshape((-1,))
        pose_ypr = poses[n, 3:6].reshape((-1,))
        pose_quat = ypr2quat(pose_ypr).reshape((-1,))

        pose = Pose()
        pose.position.x = pose_trans[0]
        pose.position.y = pose_trans[1]
        pose.position.z = pose_trans[2]
        pose.orientation.x = pose_quat[0]
        pose.orientation.y = pose_quat[1]
        pose.orientation.z = pose_quat[2]
        pose.orientation.w = pose_quat[3]
        arrow_marker.pose = pose

        # when list is used, color needs to be 1.0 not 255, such a bug!
        arrow_marker.color.r, arrow_marker.color.g, arrow_marker.color.b = (1.0, 0.0, 0)
        arrow_marker.color.a = 1.0
        arrow_marker.scale.x, arrow_marker.scale.y, arrow_marker.scale.z = (scale_size * 5, scale_size, scale_size)

        marker_array.markers.append(arrow_marker)

    # publish marker_array
    marker_pub.publish(marker_array)
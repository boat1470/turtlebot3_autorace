#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""arx_mission's detect_lane.

Same remappings and the same topic names as
turtlebot3_autorace_detect/launch/detect_lane.launch.py, so it is a drop-in
replacement - run this one instead of that one.

What the fork changes is listed at the top of detect_lane.py. The one worth
knowing here is that make_lane could reach its publish with no centre line
computed, which killed the node mid-course and left the robot driving on
whatever velocity it had last been given.

Arguments:
    calibration_mode:=true   the original's flag: registers nothing extra now
                             (the parameter callback is always registered),
                             but is still passed through so the same command
                             line works
    debug_image:=true        publish the white and yellow mask images on
                             /detect/image_white_lane_marker and
                             /detect/image_yellow_lane_marker. In the original
                             this was welded to calibration_mode, which also
                             switched the lightness auto-adjust over - so
                             looking at a mask changed what was being looked
                             at.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    param_file = os.path.join(
        get_package_share_directory('arx_mission'), 'param', 'lane.yaml')

    calibration_mode = LaunchConfiguration('calibration_mode')
    debug_image = LaunchConfiguration('debug_image')

    args = [
        DeclareLaunchArgument(
            'calibration_mode', default_value='False',
            description='Mode type [calibration, action]'),
        DeclareLaunchArgument(
            'debug_image', default_value='False',
            description='Publish the white and yellow mask images.'),
    ]

    detect_lane_node = Node(
        package='arx_mission',
        executable='detect_lane',
        name='arx_detect_lane',
        output='screen',
        parameters=[
            param_file,
            {'is_detection_calibration_mode': ParameterValue(
                calibration_mode, value_type=bool),
             'lane.publish_debug_image': ParameterValue(
                 debug_image, value_type=bool)},
        ],
        remappings=[
            ('/detect/image_input', '/camera/image_projected'),
            ('/detect/image_input/compressed', '/camera/image_projected/compressed'),
            ('/detect/image_output', '/detect/image_lane'),
            ('/detect/image_output/compressed', '/detect/image_lane/compressed'),
            ('/detect/image_output_sub1', '/detect/image_white_lane_marker'),
            ('/detect/image_output_sub1/compressed', '/detect/image_white_lane_marker/compressed'),
            ('/detect/image_output_sub2', '/detect/image_yellow_lane_marker'),
            ('/detect/image_output_sub2/compressed', '/detect/image_yellow_lane_marker/compressed')
        ]
    )

    return LaunchDescription(args + [detect_lane_node])

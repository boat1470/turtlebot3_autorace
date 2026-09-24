#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""arx_mission's intersection sign detector.

Same remappings as turtlebot3_autorace_detect/launch/detect_sign.launch.py,
minus its mission argument: that launch file builds the executable name out
of the argument so one file can start any of the five sign detectors, and
this package has forked only the intersection one. The other four can have
their own launch file when their mission is written.

The detector stays idle until mission_control arms it on /arx/armed_mission,
so starting it early costs nothing. always_on:=true runs it regardless, which
is how to tune it on its own:

    ros2 launch arx_mission detect_sign.launch.py always_on:=true
    ros2 run rqt_image_view rqt_image_view    # /detect/image_traffic_sign/compressed

Do not leave always_on set for a real run. The rules allow signs and props on
the course that belong to no mission, and a detector that is always looking
will find one of them.
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
        get_package_share_directory('arx_mission'), 'param', 'sign.yaml')

    always_on = LaunchConfiguration('always_on')

    args = [
        DeclareLaunchArgument(
            'always_on', default_value='false',
            description='Run the detector without waiting to be armed, for '
                        'tuning it on its own.'),
    ]

    detect_sign_node = Node(
        package='arx_mission',
        executable='detect_intersection_sign',
        name='arx_detect_intersection_sign',
        output='screen',
        parameters=[
            param_file,
            {'sign.always_on': ParameterValue(always_on, value_type=bool)},
        ],
        remappings=[
            # The compensated image, not the projected one: the sign stands
            # above the ground plane, so the bird's-eye view does not contain
            # it. Same input as the traffic light detector.
            ('/detect/image_input', '/camera/image_compensated'),
            ('/detect/image_input/compressed', '/camera/image_compensated/compressed'),
            ('/detect/image_output', '/detect/image_traffic_sign'),
            ('/detect/image_output/compressed', '/detect/image_traffic_sign/compressed'),
        ]
    )

    return LaunchDescription(args + [detect_sign_node])

#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""The level crossing bar detector.

Reports whether the bar is up, down, or not in view, on /detect/level_bar,
and how far away it is on /detect/level_range. It decides nothing -
mission_control reads the report and chooses when to creep and when to
stand - so this node never touches /cmd_vel, /avoid_control or /avoid_active.

It stays idle until mission_control arms it on /arx/armed_mission, so
starting it early costs nothing. always_on:=true runs it regardless, which is
how to look at the reading on its own:

    ros2 launch arx_mission detect_level.launch.py always_on:=true
    ros2 run rqt_image_view rqt_image_view    # /detect/image_level/compressed

The picture shows both sets of blobs: grey for everything the detector saw,
yellow for the ones it decided belong to the bar, and the blue line is what
was fitted through them. A wrong call can be read straight off it.

Do not leave always_on set for a real run. There is red on other props on
this course, and a detector that is always looking will find some.
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
        get_package_share_directory('arx_mission'), 'param', 'level.yaml')

    always_on = LaunchConfiguration('always_on')

    args = [
        DeclareLaunchArgument(
            'always_on', default_value='false',
            description='Run the detector without waiting to be armed, for '
                        'looking at its reading on its own.'),
    ]

    detect_level_node = Node(
        package='arx_mission',
        executable='detect_level',
        name='arx_detect_level',
        output='screen',
        parameters=[
            param_file,
            {'level.always_on': ParameterValue(always_on, value_type=bool)},
        ],
        remappings=[
            # The compensated frame, not the projected one: this reads the
            # bar's shape in the camera's own view, and the bird's-eye
            # projection is a ground plane - it stretches anything standing up
            # off the ground beyond recognition.
            ('/detect/image_input', '/camera/image_compensated'),
            ('/detect/image_output/compressed', '/detect/image_level/compressed'),
        ]
    )

    return LaunchDescription(args + [detect_level_node])

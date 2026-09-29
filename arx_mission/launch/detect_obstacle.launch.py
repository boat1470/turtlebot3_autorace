#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""The construction-zone obstacle detector.

Reports which half of the corridor ahead has a board in it, on
/detect/obstacle. It decides nothing - mission_control reads the report and
chooses which line to hug - so this node never touches /cmd_vel,
/avoid_control or /avoid_active.

It stays idle until mission_control arms it on /arx/armed_mission, so
starting it early costs nothing. always_on:=true runs it regardless, which is
how to look at the detection windows on their own:

    ros2 launch arx_mission detect_obstacle.launch.py always_on:=true
    ros2 run rqt_image_view rqt_image_view    # /detect/image_obstacle/compressed

Do not leave always_on set for a real run. The rules allow props on the
course that belong to no mission, and a detector that is always looking will
find one of them.

There is no image input to remap. /scan comes straight from the LiDAR, and
the picture this node publishes is drawn from the scan rather than from a
camera frame.
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
        get_package_share_directory('arx_mission'), 'param', 'obstacle.yaml')

    always_on = LaunchConfiguration('always_on')

    args = [
        DeclareLaunchArgument(
            'always_on', default_value='false',
            description='Run the detector without waiting to be armed, for '
                        'looking at its windows on their own.'),
    ]

    detect_obstacle_node = Node(
        package='arx_mission',
        executable='detect_obstacle',
        name='arx_detect_obstacle',
        output='screen',
        parameters=[
            param_file,
            {'obstacle.always_on': ParameterValue(always_on, value_type=bool)},
        ],
        remappings=[
            ('/detect/image_output/compressed', '/detect/image_obstacle/compressed'),
        ]
    )

    return LaunchDescription(args + [detect_obstacle_node])

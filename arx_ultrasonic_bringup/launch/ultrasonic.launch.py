#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""The level crossing bar, read with the HC-SR04 on the front of the robot.

Runs on the Pi - the sensor is wired to its GPIO - and publishes only
/detect/level_range, the distance; mission_control decides from it. The
camera detector (detect_level.launch.py) publishes /detect/level_range too,
so the two still cannot run side by side until one of them is renamed.

It stays idle until mission_control arms it on /arx/armed_mission.
always_on:=true measures regardless, which is how to check the wiring:

    ros2 launch arx_ultrasonic_bringup ultrasonic.launch.py always_on:=true
    ros2 topic echo /detect/level_range
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
        get_package_share_directory('arx_ultrasonic_bringup'), 'param', 'ultrasonic.yaml')

    always_on = LaunchConfiguration('always_on')

    args = [
        DeclareLaunchArgument(
            'always_on', default_value='false',
            description='Measure without waiting to be armed, for looking at '
                        'the reading on its own.'),
    ]

    node = Node(
        package='arx_ultrasonic_bringup',
        executable='detect_level_ultrasonic',
        name='arx_detect_level_ultrasonic',
        output='screen',
        parameters=[
            param_file,
            {'level.always_on': ParameterValue(always_on, value_type=bool)},
        ],
    )

    return LaunchDescription(args + [node])

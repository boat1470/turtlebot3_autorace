#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""The sequencer, on its own.

It used to be started inside traffic_light.launch.py, which was reasonable
while the traffic light was the only mission it ran. It is not the traffic
light's any more - it drives the intersection too, and will drive the other
four - so it lives here, and every launch file in this package now starts
exactly one thing.

What that buys: the light detector can be stopped while debugging the
intersection without also stopping the node that arms the sign detector, and
the running order below reads as what it is.

    ros2 launch turtlebot3_gazebo turtlebot3_autorace_2020.launch.py
    ros2 launch turtlebot3_autorace_camera intrinsic_camera_calibration.launch.py
    ros2 launch turtlebot3_autorace_camera extrinsic_camera_calibration.launch.py
    ros2 launch arx_mission mission_control.launch.py     <- this one
    ros2 launch arx_mission traffic_light.launch.py
    ros2 launch arx_mission detect_sign.launch.py
    ros2 launch arx_mission detect_lane.launch.py
    ros2 launch arx_mission control_lane.launch.py

This one comes before control_lane so the /arx/drive_enable gate is already
being published when control_lane appears. control_lane starts held either
way, so the order is a courtesy rather than a requirement.
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
        get_package_share_directory('arx_mission'), 'param', 'mission.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    start_stage = LaunchConfiguration('start_stage')

    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Follow /clock. Both give-up timers are measured on this.'),
        DeclareLaunchArgument(
            'start_stage', default_value='wait_green',
            description='Stage to begin in. Anything but wait_green skips the '
                        'missions before it. full.launch.py start:=<preset> '
                        'sets this together with where the robot is standing.'),
    ]

    mission_control = Node(
        package='arx_mission',
        executable='mission_control',
        name='arx_mission_control',
        output='screen',
        parameters=[
            param_file,
            # A LaunchConfiguration evaluates to a string, and whether that
            # reaches a bool parameter as True or as a type error depends on
            # the launch_ros version, so the type is stated.
            {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
            # After param_file, so it wins. mission.yaml deliberately does not
            # carry start_stage: a run that skips missions is decided on the
            # command line for one run, not left set in a file where the next
            # person inherits it without knowing.
            {'start_stage': ParameterValue(start_stage, value_type=str)},
        ],
    )

    return LaunchDescription(args + [mission_control])

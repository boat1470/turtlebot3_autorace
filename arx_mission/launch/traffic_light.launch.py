#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Traffic light mission: the detector plus the sequencer that acts on it.

Expects the camera pipeline and lane following to be up already:

    ros2 launch turtlebot3_gazebo turtlebot3_autorace_2020.launch.py
    ros2 launch turtlebot3_autorace_camera intrinsic_camera_calibration.launch.py
    ros2 launch turtlebot3_autorace_camera extrinsic_camera_calibration.launch.py
    ros2 launch turtlebot3_autorace_detect detect_lane.launch.py
    ros2 launch turtlebot3_autorace_mission control_lane.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    param_dir = os.path.join(get_package_share_directory('arx_mission'), 'param')
    # One file per node. All three detectors declare frame_skip and
    # publish_debug_image and want different values, and a /**: key hands
    # every node in a file the same value.
    light_param_file = os.path.join(param_dir, 'light.yaml')
    mission_param_file = os.path.join(param_dir, 'mission.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    always_on = LaunchConfiguration('always_on')

    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Follow /clock. The give-up timer is measured on this.'),
        DeclareLaunchArgument(
            'always_on', default_value='false',
            description='Run the detector without waiting to be armed, for '
                        'tuning thresholds on their own.'),
    ]

    # Wrapped in ParameterValue with an explicit type. A LaunchConfiguration
    # evaluates to a string, and every launch file in this repository passes
    # one straight into a parameter dict; whether that reaches a bool
    # parameter as True or as a type error depends on the launch_ros version.
    def typed(value, value_type):
        return ParameterValue(value, value_type=value_type)

    detect_traffic_light = Node(
        package='arx_mission',
        executable='detect_traffic_light',
        name='arx_light_detector',
        output='screen',
        parameters=[
            light_param_file,
            {'use_sim_time': typed(use_sim_time, bool),
             'always_on': typed(always_on, bool)},
        ],
        remappings=[
            # The compensated image, not the projected one: the lamp is above
            # the ground plane, so the bird's-eye view does not contain it.
            ('/detect/image_input', '/camera/image_compensated'),
        ],
    )

    mission_control = Node(
        package='arx_mission',
        executable='mission_control',
        name='arx_mission_control',
        output='screen',
        parameters=[
            mission_param_file,
            {'use_sim_time': typed(use_sim_time, bool)},
        ],
    )

    return LaunchDescription(args + [detect_traffic_light, mission_control])

#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""The traffic light detector.

The sequencer used to be started here as well. It is in
mission_control.launch.py now - it runs the intersection too, and will run the
other four missions, so it was not the traffic light's to start.

This detector does nothing until mission_control arms it, so on its own it is
only useful with always_on:=true, for tuning the thresholds:

    ros2 launch arx_mission traffic_light.launch.py always_on:=true
    ros2 run rqt_reconfigure rqt_reconfigure     # /arx_light_detector
    ros2 run rqt_image_view rqt_image_view       # /arx/image_traffic_light/compressed

Never leave always_on set for a real run. The rules allow props on the course
that belong to no mission, and a detector that is always looking will stop the
robot for one of them.

Expects the camera pipeline to be up already. See mission_control.launch.py
for the full running order, or use full.launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # One file per node. All three detectors declare frame_skip and
    # publish_debug_image and want different values, and a /**: key hands
    # every node in a file the same value.
    param_file = os.path.join(
        get_package_share_directory('arx_mission'), 'param', 'light.yaml')

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
            param_file,
            {'use_sim_time': typed(use_sim_time, bool),
             'always_on': typed(always_on, bool)},
        ],
        remappings=[
            # The compensated image, not the projected one: the lamp is above
            # the ground plane, so the bird's-eye view does not contain it.
            ('/detect/image_input', '/camera/image_compensated'),
        ],
    )

    return LaunchDescription(args + [detect_traffic_light])

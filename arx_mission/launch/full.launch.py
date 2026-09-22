#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Whole stack in one command.

    . notes/env.sh
    ros2 launch arx_mission full.launch.py

Arguments:
    sim:=false          attach to an already running simulator, or to the
                        real robot
    calibration:=true   bring the detectors up in calibration mode, which is
                        the only way their parameter callbacks get registered
                        and therefore the only way rqt_reconfigure can change
                        a threshold live

Stages are separated by TimerAction because each one needs the topic the
previous one publishes: the camera pipeline has nothing to rectify until the
bridge is up, and detect_lane has nothing to threshold until the projection
is running. Nodes started too early do not crash, they just sit there, which
looks exactly like a broken pipeline.

The one stage whose order is not about topics is the sequencer: it is started
before control_lane so that nothing is ever driving unheld. See the comment
on the LaunchDescription below.

There is deliberately no gui argument. turtlebot3_autorace_2020.launch.py adds
gzclient unconditionally and takes no argument for it, so offering the option
here would be a lie. Run the simulator separately if a headless one is needed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def include(package, launch_file, condition=None, **kwargs):
    src = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(package), 'launch', launch_file))
    return IncludeLaunchDescription(
        src, condition=condition, launch_arguments=list(kwargs.items()) or None)


def generate_launch_description():
    sim = LaunchConfiguration('sim')
    calibration = LaunchConfiguration('calibration')

    args = [
        DeclareLaunchArgument('sim', default_value='true',
                              description='Start the Gazebo simulator too.'),
        DeclareLaunchArgument('calibration', default_value='false',
                              description='Detectors in calibration mode, so '
                                          'rqt_reconfigure can retune them live.'),
    ]

    simulator = include('turtlebot3_gazebo', 'turtlebot3_autorace_2020.launch.py',
                        condition=IfCondition(sim))

    camera = [
        include('turtlebot3_autorace_camera', 'intrinsic_camera_calibration.launch.py'),
        include('turtlebot3_autorace_camera', 'extrinsic_camera_calibration.launch.py',
                calibration_mode=calibration),
    ]

    lane = [
        include('turtlebot3_autorace_detect', 'detect_lane.launch.py',
                calibration_mode=calibration),
        # ours, not turtlebot3_autorace_mission's: this one starts held
        include('arx_mission', 'control_lane.launch.py'),
    ]

    mission = include('arx_mission', 'traffic_light.launch.py')

    # The sequencer still comes up before control_lane. It no longer has to -
    # this package's control_lane starts held and cannot move until
    # /arx/drive_enable says so - but this way the gate is already being
    # published when control_lane appears, so it never publishes the zero
    # Twist of a hold it was going to be told about anyway.
    return LaunchDescription(args + [
        simulator,
        TimerAction(period=8.0, actions=camera),
        TimerAction(period=14.0, actions=[mission]),
        TimerAction(period=16.0, actions=lane),
    ])

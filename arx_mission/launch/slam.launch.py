#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Map the course with Cartographer, at 0.025 m (ARX).

Runs on the PC, never the Pi - the Pi has no Cartographer. Only robot
bringup has to be up on the Pi, and full.launch has to be STOPPED: its
control_lane holds /cmd_vel at zero and the teleop cannot move the robot.

    ros2 launch arx_mission slam.launch.py
    ros2 run turtlebot3_teleop teleop_keyboard     # TURTLEBOT3_MODEL=burger

Drive slowly - turning fast is what doubles walls. Through the tunnel and
back once or twice. Then, with this still running:

    mkdir -p ~/turtlebot3_ws/src/turtlebot3_autorace/arx_mission/map
    ros2 run nav2_map_server map_saver_cli \\
        -f ~/turtlebot3_ws/src/turtlebot3_autorace/arx_mission/map/tunnel

and rebuild arx_mission so nav.launch.py picks the new map up.

0.025 rather than turtlebot3_cartographer's 0.05 because the tunnel's way in
is about 0.30 m: at 0.05 each wall edge can land a whole cell off, and the
gap can come out at 0.25.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    args = [
        DeclareLaunchArgument('resolution', default_value='0.025',
                              description='Map cell size, metres.'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
    ]
    cartographer = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('turtlebot3_cartographer'),
            'launch', 'cartographer.launch.py')),
        launch_arguments={
            'resolution': LaunchConfiguration('resolution'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items())
    return LaunchDescription(args + [cartographer])

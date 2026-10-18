#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Nav2 on the tunnel map, with RViz (ARX).

Runs on the PC, never the Pi. Only robot bringup has to be up on the Pi.

    ros2 launch arx_mission nav.launch.py
    ros2 launch arx_mission nav.launch.py map:=/path/to/other.yaml rviz:=false

STOP full.launch FIRST. Its control_lane publishes a zero Twist on /cmd_vel
every tick while it is holding, and fights every command Nav2 sends.

Then, in RViz: 2D Pose Estimate where the robot stands (the red scan should
land on the walls), and Nav2 Goal at the exit. To check the gap is open, add
a Map display on /global_costmap/costmap with the costmap colour scheme - the
way in has to show a thin channel, not a solid wall.

The params are param/nav2.yaml; param/nav2_wrg2026.yaml is what they started
from, kept for comparison.

full.launch.py starts this too (nav:=true, the default without the
simulator) with rviz:=false and cmd_vel_out:=/arx/cmd_vel_nav, so that Nav2's
commands go through control_lane, which passes them on to /cmd_vel only
while mission_control says Nav2 has the wheel. Started on its own, Nav2
publishes /cmd_vel itself.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    share = get_package_share_directory('arx_mission')
    map_yaml = LaunchConfiguration('map')
    params = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    cmd_vel_out = LaunchConfiguration('cmd_vel_out')

    args = [
        DeclareLaunchArgument(
            'map', default_value=os.path.join(share, 'map', 'tunnel.yaml'),
            description='Map to localise on.'),
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(share, 'param', 'nav2.yaml'),
            description='Nav2 parameters.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Follow /clock - only with the simulator.'),
        DeclareLaunchArgument(
            'rviz', default_value='true', description='Start RViz too.'),
        DeclareLaunchArgument(
            'cmd_vel_out', default_value='/cmd_vel',
            description='Where collision_monitor sends the final command. '
                        '/cmd_vel drives the robot directly; full.launch.py '
                        'passes /arx/cmd_vel_nav for control_lane to forward.'),
    ]

    params_out = RewrittenYaml(
        source_file=params,
        param_rewrites={'cmd_vel_out_topic': cmd_vel_out},
        convert_types=True)

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')),
        launch_arguments={
            'map': map_yaml,
            'params_file': params_out,
            'use_sim_time': use_sim_time,
            'autostart': 'true',
        }.items())

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', os.path.join(
            get_package_share_directory('turtlebot3_navigation2'), 'rviz', 'tb3_navigation2.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz))

    return LaunchDescription(args + [nav2, rviz_node])

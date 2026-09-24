#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""arx_mission's control_lane: the ROBOTIS node with a drive-enable gate.

Same remappings as turtlebot3_autorace_mission/launch/control_lane.launch.py,
so it is a drop-in replacement - run this one instead of that one.

The difference is that this node starts stopped and waits for
/arx/drive_enable, which mission_control publishes. The original starts with
avoid_active false and MAX_VEL 0.1 and drives from its first lane message,
which leaves nothing able to hold it during the seconds DDS takes to match
subscriptions.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    param_file = os.path.join(
        get_package_share_directory('arx_mission'), 'param', 'mission.yaml')

    control_lane = Node(
        package='arx_mission',
        executable='control_lane',
        name='arx_control_lane',
        output='screen',
        parameters=[param_file],
        remappings=[
            ('/control/lane', '/detect/lane'),
            ('/control/cmd_vel', '/cmd_vel'),
        ],
    )

    return LaunchDescription([control_lane])

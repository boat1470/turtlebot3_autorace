#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Whole stack in one command.

    . notes/env.sh
    ros2 launch arx_mission full.launch.py                  # the whole course
    ros2 launch arx_mission full.launch.py start:=mission3  # one mission only

Arguments:
    start:=<preset>     where a run begins. `full` is the start line with
                        every mission ahead of it, and is the default. Any
                        other preset stands the robot where that mission
                        begins and tells the sequencer to start in its stage,
                        so the missions before it never run. See PRESETS.
    sim:=false          attach to an already running simulator, or to the
                        real robot
    calibration:=true   bring detect_lane up in calibration mode, which turns
                        its yellow lightness auto-adjust on and its white one
                        off. It no longer has anything to do with whether
                        rqt_reconfigure works - this package's detectors
                        register their parameter callbacks in every mode.
    debug_image:=true   publish the lane mask images

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
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Where a run may begin (ARX). Each preset is a place to stand and the stage
# to start in, because either alone is useless: a sequencer told to start at
# the construction sign while the robot is on the start line will drive the
# whole course in the wrong mission, and a robot teleported to the sign while
# the sequencer waits for a traffic light will sit there.
#
# `x` and `y` reach turtlebot3_autorace_2020.launch.py through
# SetLaunchConfiguration rather than launch_arguments, because that file
# declares no launch arguments at all - it only reads
# LaunchConfiguration('x_pose', default='0.8'), which picks up whatever this
# scope has already set.
#
# There is no yaw. spawn_turtlebot3.launch.py passes -x -y -z to
# `ros_gz_sim create` and nothing else, so the yaw_pose the example sets has
# never had any effect and the robot always spawns facing +x. Measured, not
# assumed. Every preset therefore has to be somewhere that facing +x is a
# reasonable heading to begin from.
PRESETS = {
    # The start line, with the traffic light ahead. The numbers are the
    # example's own defaults, repeated so that `full` is described in the
    # same place as the rest rather than by their absence.
    'full': {
        'stage': 'wait_green',
        'x': '0.8',
        'y': '-1.747',
    },
    # Where a full run ends up once the construction board has been read:
    # measured at (0.581, 0.252) facing +1.6 degrees. Close enough to the
    # board to confirm it within a second or two, which is the point - the
    # approach is not what is being worked on, the avoidance after it is.
    'mission3': {
        'stage': 'drive_to_construction',
        'x': '0.58',
        'y': '0.25',
    },
}


def include(package, launch_file, condition=None, **kwargs):
    src = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(package), 'launch', launch_file))
    return IncludeLaunchDescription(
        src, condition=condition, launch_arguments=list(kwargs.items()) or None)


def stack(context, *unused_args, **unused_kwargs):
    """Build the stack once `start` has a value to look up (ARX).

    An OpaqueFunction because the preset has to be chosen from a dict, and a
    LaunchConfiguration is a substitution rather than a string until a context
    exists to resolve it in.
    """
    start = LaunchConfiguration('start').perform(context)
    if start not in PRESETS:
        # Refuse rather than fall back to `full`. Falling back would run the
        # whole course and look like the preset had been applied, which is
        # the slow failure this argument exists to avoid.
        raise RuntimeError(
            f'start:={start} is not a preset. '
            f'Known: {", ".join(sorted(PRESETS))}')
    preset = PRESETS[start]

    sim = LaunchConfiguration('sim')
    calibration = LaunchConfiguration('calibration')
    debug_image = LaunchConfiguration('debug_image')

    simulator = include('turtlebot3_gazebo', 'turtlebot3_autorace_2020.launch.py',
                        condition=IfCondition(sim))

    camera = [
        include('turtlebot3_autorace_camera', 'intrinsic_camera_calibration.launch.py'),
        include('turtlebot3_autorace_camera', 'extrinsic_camera_calibration.launch.py',
                calibration_mode=calibration),
    ]

    lane = [
        # ours, not turtlebot3_autorace_detect's: this one survives losing
        # both lane lines at once
        include('arx_mission', 'detect_lane.launch.py',
                calibration_mode=calibration, debug_image=debug_image),
        # ours, not turtlebot3_autorace_mission's: this one starts held
        include('arx_mission', 'control_lane.launch.py'),
    ]

    mission = [
        include('arx_mission', 'mission_control.launch.py',
                start_stage=preset['stage']),
        # Both detectors idle until mission_control arms them, so their
        # position in the order only has to be after the camera pipeline.
        include('arx_mission', 'traffic_light.launch.py'),
        include('arx_mission', 'detect_sign.launch.py'),
        include('arx_mission', 'detect_obstacle.launch.py'),
    ]

    # The sequencer still comes up before control_lane. It no longer has to -
    # this package's control_lane starts held and cannot move until
    # /arx/drive_enable says so - but this way the gate is already being
    # published when control_lane appears, so it never publishes the zero
    # Twist of a hold it was going to be told about anyway.
    #
    # Within the mission stage the sequencer is first for the same reason: it
    # is what arms the detectors beside it.
    return [
        # Before the simulator, so the spawn reads them.
        SetLaunchConfiguration('x_pose', preset['x']),
        SetLaunchConfiguration('y_pose', preset['y']),
        simulator,
        TimerAction(period=8.0, actions=camera),
        TimerAction(period=14.0, actions=mission),
        TimerAction(period=16.0, actions=lane),
    ]


def generate_launch_description():
    args = [
        DeclareLaunchArgument('start', default_value='full',
                              description='Preset to begin from: '
                                          + ', '.join(sorted(PRESETS))),
        DeclareLaunchArgument('sim', default_value='true',
                              description='Start the Gazebo simulator too.'),
        DeclareLaunchArgument('calibration', default_value='false',
                              description='detect_lane in calibration mode.'),
        DeclareLaunchArgument('debug_image', default_value='false',
                              description='Publish the lane mask images.'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=stack)])

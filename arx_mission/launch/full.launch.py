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

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

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
# `yaw` is in degrees, and is not part of the spawn. spawn_turtlebot3.
# launch.py passes -x -y -z to `ros_gz_sim create` and nothing else, so the
# yaw_pose the example sets has never had any effect and the robot always
# comes up facing +x. A preset that needs another heading is turned on the
# spot afterwards, with gz's own set_pose service - see turn_robot below.
#
# That turn is invisible to odometry: gz's DiffDrive integrates from zero
# whatever pose the model is in, so /odom keeps reading the heading the robot
# was spawned at. Measured - rotating the model from yaw -3.130 to 0.000 left
# /odom's quaternion identical to twelve decimal places. mission_control's
# heading gates are world angles, so the same number is handed to it as
# spawn_yaw_deg and added back on.
PRESETS = {
    # The start line, with the traffic light ahead. The numbers are the
    # example's own defaults, repeated so that `full` is described in the
    # same place as the rest rather than by their absence.
    'full': {
        'stage': 'wait_green',
        'x': '0.8',
        'y': '-1.747',
        'yaw': '0.0',
    },
    # Where a full run ends up once the construction board has been read:
    # measured at (0.581, 0.252) facing +1.6 degrees. Close enough to the
    # board to confirm it within a second or two, which is the point - the
    # approach is not what is being worked on, the avoidance after it is.
    'mission3': {
        'stage': 'drive_to_construction',
        'x': '0.58',
        'y': '0.25',
        'yaw': '0.0',
    },
    # Where a mission3 run ends up once the parking board has been read and
    # the robot has stopped at it: measured at (1.2797, 1.7449) facing
    # -179.3 degrees. That is the middle of the lane - white at y 1.879,
    # yellow at 1.625 - with 0.54 m still to go to the board at x 0.74 and
    # 0.77 m to the mouth of the lot at x 0.506.
    #
    # It starts in drive_to_lot rather than drive_to_parking, which is where
    # the robot was standing: drive_to_parking is mission 3's last stage, and
    # beginning there would hunt the parking board a second time. The point
    # of the preset is to be past that.
    #
    # It is the first preset that needs a heading of its own. Everything up
    # to here has been driven in +x.
    'mission4': {
        'stage': 'drive_to_lot',
        'x': '1.28',
        'y': '1.745',
        'yaw': '180.0',
    },
}


# Where the two parking bays are (ARX), measured off the course texture the
# same way as everything else in this mission - see notes/track_geometry.py.
# The pocket runs x 0.135 to 0.877 and y 0.500 to 1.008, split into three
# columns by dashed lines at x 0.381 and 0.631: a bay, the way in, a bay.
#
# Named from the ROBOT's seat, not the course's, so that `blocked:=right` and
# the `right taken` that comes back on /detect/obstacle_side are the same
# side. The robot drives in facing -y, so its left is +x: the bay at x 0.754
# is the one on its left, and the one at x 0.258 is on its right. Nothing in
# this package names a side after the course - that is the mistake that once
# drove the construction stage into the sign standing beside the track.
#
# It does mean these are only the robot's left and right while it comes in
# the way this mission does. Anything approaching the pocket from the other
# end would have them the other way round.
BAYS = {
    'left': ('0.754', '0.754'),
    'right': ('0.258', '0.754'),
}


def block_bay(which):
    """Put a robot in one of the bays, so the free one has to be found (ARX).

    The simulator ships neither bay occupied, so `which bay is free` has only
    ever had one answer there and any rule at all would look like it worked.

    Spawned here rather than added to turtlebot3_autorace_2020.world, which
    belongs to the example: this leaves that file alone, and it means both
    cases can be run without editing anything.
    """
    x, y = BAYS[which]
    model = os.path.join(get_package_share_directory('arx_mission'),
                         'model', 'parking_dummy.sdf')
    return Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-file', model, '-name', f'parking_dummy_{which}',
                   '-x', x, '-y', y, '-z', '0.0'])


def turn_robot(preset):
    """Stand the robot on the preset's heading, after it has spawned (ARX).

    `ros_gz_sim create` is given no yaw by the example's spawn, so the only
    way to face another direction is to turn the model afterwards. gz's
    set_pose service takes a whole pose, so the position goes back in
    unchanged - the spawn has already put it there, and repeating it means
    the robot cannot creep away while the stack is still coming up.

    It is retried because there is no ordering between this and the spawn
    that launch can express: the service exists as soon as the server does,
    and answers `false` until the model is in the world. Reading the pose
    back is what says it worked; a run that started on the wrong heading
    would otherwise look like a driving fault.
    """
    world = 'default'  # <world name='default'> in turtlebot3_autorace_2020.world
    name = os.environ.get('TURTLEBOT3_MODEL', 'burger_cam')
    yaw = math.radians(float(preset['yaw']))
    script = f'''
for i in $(seq 1 20); do
  gz service -s /world/{world}/set_pose \\
    --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 \\
    --req 'name: "{name}", position: {{x: {preset['x']}, y: {preset['y']}, z: 0.01}},
           orientation: {{z: {math.sin(yaw / 2):.9f}, w: {math.cos(yaw / 2):.9f}}}' \\
    2>/dev/null | grep -q 'data: true' && {{
      echo "[arx] standing {name} at ({preset['x']}, {preset['y']}) facing {preset['yaw']} deg"
      gz model -m {name} -p 2>/dev/null | tail -2
      exit 0
  }}
  sleep 1
done
echo "[arx] COULD NOT TURN {name} - the run is on the wrong heading" >&2
exit 1
'''
    return ExecuteProcess(cmd=['bash', '-c', script], output='screen')


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

    blocked = LaunchConfiguration('blocked').perform(context)
    if blocked not in BAYS and blocked != 'none':
        raise RuntimeError(
            f'blocked:={blocked} is not a bay. '
            f'Known: none, {", ".join(sorted(BAYS))}')

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
                start_stage=preset['stage'],
                spawn_yaw_deg=preset['yaw']),
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
        # Only when there is something to turn, so a preset that faces +x
        # runs exactly as it did before this existed - the spawn already
        # puts the robot there, and a teleport that changes nothing is still
        # a teleport. After the spawn and well before anything can drive:
        # control_lane is not up until 16 s, and starts held even then.
        *([TimerAction(period=6.0, actions=[turn_robot(preset)])]
          if float(preset['yaw']) else []),
        # After the robot's own spawn, so the two creates cannot race for the
        # same world.
        *([TimerAction(period=7.0, actions=[block_bay(blocked)])]
          if blocked != 'none' else []),
        TimerAction(period=8.0, actions=camera),
        TimerAction(period=14.0, actions=mission),
        TimerAction(period=16.0, actions=lane),
    ]


def generate_launch_description():
    args = [
        DeclareLaunchArgument('start', default_value='full',
                              description='Preset to begin from: '
                                          + ', '.join(sorted(PRESETS))),
        DeclareLaunchArgument('blocked', default_value='none',
                              description='Park a robot in one of the parking '
                                          'bays: none, '
                                          + ', '.join(sorted(BAYS))),
        DeclareLaunchArgument('sim', default_value='true',
                              description='Start the Gazebo simulator too.'),
        DeclareLaunchArgument('calibration', default_value='false',
                              description='detect_lane in calibration mode.'),
        DeclareLaunchArgument('debug_image', default_value='false',
                              description='Publish the lane mask images.'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=stack)])

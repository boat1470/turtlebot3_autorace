#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""How long each part of a recorded run took.

    python3 calibrate_ui/run_times.py                 # the newest run in ~/runs
    python3 calibrate_ui/run_times.py ~/runs/1017_2240

Reads a bag made with lane_debug's REC button. The timeline comes from
mission_control's own log lines on /rosout - every `stage A -> B: why` - so
nothing extra has to be recorded, and the times are the robot's clock, not
when the bag happened to write.

Prints each stage with when it started and how long it lasted, then the
run's total from the first time the robot moved (/cmd_vel not zero) to the
last stage change. A run started from a preset (mission4 ...) is timed from
there, not from the start line.
"""

import os
import re
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

RUNS = os.path.expanduser('~/runs')
STAGE = re.compile(r'^stage (\S+) -> (\S+): (.*)$')
START = re.compile(r'^starting at stage (\S+)')


def newest_run():
    runs = [os.path.join(RUNS, d) for d in os.listdir(RUNS)
            if os.path.isfile(os.path.join(RUNS, d, 'metadata.yaml'))]
    if not runs:
        sys.exit(f'no recorded runs in {RUNS}')
    return max(runs, key=os.path.getmtime)


def read(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {'/rosout', '/cmd_vel'} & set(types)
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted)))
    classes = {t: get_message(types[t]) for t in wanted}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, deserialize_message(data, classes[topic]), t_ns * 1e-9


def main():
    path = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else newest_run()
    events = []          # (time, stage entered, why)
    first_move = None
    first_stage = None
    for topic, msg, t in read(path):
        if topic == '/cmd_vel':
            twist = getattr(msg, 'twist', msg)       # TwistStamped or Twist
            if first_move is None and (abs(twist.linear.x) > 1e-3 or abs(twist.angular.z) > 1e-3):
                first_move = t
            continue
        if msg.name != 'arx_mission_control':
            continue
        when = msg.stamp.sec + msg.stamp.nanosec * 1e-9
        m = STAGE.match(msg.msg)
        if m:
            if first_stage is None:
                first_stage = (when, m.group(1))
            events.append((when, m.group(2), m.group(3)))
            continue
        m = START.match(msg.msg)
        if m and first_stage is None:
            first_stage = (when, m.group(1))

    print(f'run: {path}')
    if first_stage is None:
        sys.exit('no mission_control stage lines on /rosout - was mission_control running?')

    t0 = first_move if first_move is not None else first_stage[0]
    timeline = [(first_stage[0], first_stage[1], 'start')] + events
    print(f'  {"at":>7}  {"took":>7}  stage')
    for i, (when, stage, why) in enumerate(timeline):
        end = timeline[i + 1][0] if i + 1 < len(timeline) else None
        took = '' if end is None else f'{end - when:6.1f}s'
        print(f'  {when - t0:6.1f}s  {took:>7}  {stage:<22} {why}')

    last = timeline[-1][0]
    print(f'total: {last - t0:.1f} s from the first move to the last stage change'
          + ('' if first_move is not None else ' (no /cmd_vel in the bag - timed from the first stage)'))


if __name__ == '__main__':
    main()

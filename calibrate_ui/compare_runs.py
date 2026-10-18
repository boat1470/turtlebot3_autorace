#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Recorded runs side by side, part by part - time, distance and speed.

    python3 calibrate_ui/compare_runs.py                # the newest 3 runs in ~/runs
    python3 calibrate_ui/compare_runs.py -n 5
    python3 calibrate_ui/compare_runs.py ~/runs/1018_1014 ~/runs/1018_1018

Parts come from mission_control's lines on /rosout (stage changes, the
tunnel sign, the Nav2 goal and result) and distance from /odom, so a run
started from mission4 is split as: mission 4 up to the 17B arc, the lot
entry, the bay, lane following to the tunnel sign, the creep to the front,
the hand-off to Nav2 and Nav2 itself. A part a run did not reach is "-".
"""

import argparse
import math
import os
import re

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

RUNS = os.path.expanduser('~/runs')
STAGE = re.compile(r'stage (\S+) -> (\S+): (.*)')
MARKS = (('tunnel sign,', 'sign'), ('at the tunnel front', 'front'),
         ('Nav2 goal', 'goal'), ('Nav2 reached', 'exit'), ('Nav2 did not', 'exit'))


def analyse(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    want = [t for t in ('/rosout', '/odom', '/cmd_vel') if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=want))
    cls = {t: get_message(types[t]) for t in want}
    stages, marks, odom = [], {}, []
    first_move, result = None, '?'
    while reader.has_next():
        topic, data, t = reader.read_next()
        t *= 1e-9
        msg = deserialize_message(data, cls[topic])
        if topic == '/cmd_vel':
            tw = getattr(msg, 'twist', msg)
            if first_move is None and abs(tw.linear.x) > 1e-3:
                first_move = t
        elif topic == '/odom':
            p = msg.pose.pose.position
            odom.append((t, p.x, p.y))
        elif msg.name == 'arx_mission_control':
            when = msg.stamp.sec + msg.stamp.nanosec * 1e-9
            m = STAGE.match(msg.msg)
            if m:
                stages.append((when, m.group(2)))
                if m.group(2) == 'stopped':
                    result = m.group(3)
            for key, name in MARKS:
                if key in msg.msg and name not in marks:
                    marks[name] = when

    def dist(a, b):
        pts = [(x, y) for t, x, y in odom if a <= t <= b]
        return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(pts, pts[1:]))

    at = {}
    for when, s in stages:
        at.setdefault(s, when)
    end = marks.get('exit') or (stages[-1][0] if stages else None)
    parts = [
        ('to the 17B arc end', first_move, at.get('drive_to_lot')),
        ('lot entry', at.get('drive_to_lot'), at.get('pick_bay')),
        ('pick bay', at.get('pick_bay'), at.get('park_in_bay')),
        ('bay: in and out', at.get('park_in_bay'), at.get('drive_to_level')),
        ('lane following to the sign', at.get('drive_to_tunnel'), marks.get('sign')),
        ('creep to the front', marks.get('sign'), marks.get('front')),
        ('hand-off to Nav2', marks.get('front'), marks.get('goal')),
        ('Nav2 driving', marks.get('goal'), marks.get('exit')),
        ('TOTAL', first_move, end),
    ]
    out = {}
    for name, a, b in parts:
        if a is not None and b is not None and b >= a:
            out[name] = (b - a, dist(a, b))
    return out, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('-n', type=int, default=3, help='newest N runs when none are named')
    a = ap.parse_args()
    if a.runs:
        paths = [os.path.expanduser(p.rstrip('/')) for p in a.runs]
    else:
        done = [os.path.join(RUNS, d) for d in os.listdir(RUNS)
                if os.path.isfile(os.path.join(RUNS, d, 'metadata.yaml'))]
        paths = sorted(done, key=lambda p: os.path.getmtime(os.path.join(p, 'metadata.yaml')))[-a.n:]
    res = [analyse(p) for p in paths]
    names = [os.path.basename(p) for p in paths]
    keys = []
    for o, _ in res:
        keys += [k for k in o if k not in keys]
    w = 20
    print(f'{"":28s}' + ''.join(f'{n:>{w}s}' for n in names))
    for k in keys:
        row = f'{k:28s}'
        for o, _ in res:
            if k in o:
                d, m = o[k]
                v = m / d if d > 0.3 else 0.0
                row += f'{d:7.1f}s {m:5.2f}m {v:4.2f}'.rjust(w)
            else:
                row += '-'.rjust(w)
        print(row)
    print(f'{"ended":28s}' + ''.join(f'{(r or "?")[:w - 1]:>{w}s}' for _, r in res))
    print('(each cell: time, distance on odom, mean speed m/s)')


if __name__ == '__main__':
    main()

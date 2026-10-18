#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Frames out of a recorded run, as one picture - to see what detect_lane fitted.

    python3 calibrate_ui/bag_frames.py                          # newest run, whole thing
    python3 calibrate_ui/bag_frames.py ~/runs/1017_2303 --from 20 --to 30
    python3 calibrate_ui/bag_frames.py --stage park_in_bay      # just that stage
    python3 calibrate_ui/bag_frames.py --topic /detect/image_yellow_lane_marker/compressed

Times are seconds from the start of the bag. --stage takes the span of a
mission_control stage off /rosout instead. The frames are spread evenly over
the span (--count of them, 16 by default), each stamped with its time, the
lane_state of the moment and the stage, and written as one image next to the
bag (<run>/frames_<from>-<to>.jpg). Nothing is published - unlike
`ros2 bag play`, which would send the recorded /cmd_vel to a robot listening
on the same ROS_DOMAIN_ID.

image_lane is the bird's-eye frame with the fits drawn on: solid pink is the
line being steered by, dashed is the centre it gives. The yellow and white
masks are only in a bag when detect_lane publishes them - full.launch
debug_image:=true.
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

RUNS = os.path.expanduser('~/runs')
STAGE = re.compile(r'^stage (\S+) -> (\S+): ')
START = re.compile(r'^starting at stage (\S+)')


def newest_run():
    runs = [os.path.join(RUNS, d) for d in os.listdir(RUNS)
            if os.path.isfile(os.path.join(RUNS, d, 'metadata.yaml'))]
    if not runs:
        sys.exit(f'no finished runs in {RUNS} (a run still recording has no metadata.yaml yet)')
    return max(runs, key=lambda p: os.path.getmtime(os.path.join(p, 'metadata.yaml')))


def open_bag(path, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    have = [t for t in topics if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=have))
    return reader, {t: get_message(types[t]) for t in have}, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?')
    ap.add_argument('--topic', default='/detect/image_lane/compressed')
    ap.add_argument('--from', dest='t_from', type=float, default=None)
    ap.add_argument('--to', dest='t_to', type=float, default=None)
    ap.add_argument('--stage', default=None, help='a mission_control stage name')
    ap.add_argument('--count', type=int, default=16)
    ap.add_argument('--width', type=int, default=320, help='width of each frame')
    a = ap.parse_args()
    path = os.path.expanduser(a.run) if a.run else newest_run()

    # First pass: when the bag starts, the stage spans and lane_state.
    reader, cls, types = open_bag(path, ['/rosout', '/detect/lane_state', a.topic])
    if a.topic not in types:
        imgs = sorted(t for t in types if 'image' in t)
        sys.exit(f'{a.topic} is not in this bag. Images it has: {", ".join(imgs) or "none"}')
    t0 = None
    stages = []                    # (time, stage)
    states = []                    # (time, lane_state)
    frame_times = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        t *= 1e-9
        t0 = t if t0 is None else t0
        if topic == a.topic:
            frame_times.append(t)
            continue
        msg = deserialize_message(data, cls[topic])
        if topic == '/detect/lane_state':
            states.append((t, msg.data))
        elif msg.name == 'arx_mission_control':
            when = msg.stamp.sec + msg.stamp.nanosec * 1e-9
            m = START.match(msg.msg) or STAGE.match(msg.msg)
            if m:
                stages.append((when, m.groups()[-1]))

    def stage_at(t):
        cur = '-'
        for when, s in stages:
            if when <= t:
                cur = s
        return cur

    def state_at(t):
        cur = None
        for when, s in states:
            if when > t:
                break
            cur = s
        return cur

    if a.stage:
        spans = [(when, stages[i + 1][0] if i + 1 < len(stages) else frame_times[-1])
                 for i, (when, s) in enumerate(stages) if s == a.stage]
        if not spans:
            sys.exit(f'stage {a.stage} not in this run. Stages: '
                     + ', '.join(dict.fromkeys(s for _, s in stages)))
        lo, hi = spans[0]
    else:
        lo = t0 + (a.t_from or 0.0)
        hi = t0 + a.t_to if a.t_to is not None else frame_times[-1]
    inside = [t for t in frame_times if lo <= t <= hi]
    if not inside:
        sys.exit('no frames in that span')
    k = min(a.count, len(inside))
    wanted = set(inside[int(i * (len(inside) - 1) / max(k - 1, 1))] for i in range(k))

    # Second pass: decode just the frames picked.
    reader, cls, _ = open_bag(path, [a.topic])
    tiles = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        t *= 1e-9
        if t not in wanted:
            continue
        msg = deserialize_message(data, cls[topic])
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (a.width, int(img.shape[0] * a.width / img.shape[1])))
        label = f'{t - t0:.1f}s ls={state_at(t)} {stage_at(t)}'
        cv2.rectangle(img, (0, 0), (a.width, 20), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(img)

    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 2, cv2.BORDER_CONSTANT) for t in tiles]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = os.path.join(path, f'frames_{lo - t0:.0f}-{hi - t0:.0f}.jpg')
    cv2.imwrite(out, sheet)
    print(f'{k} frames, {lo - t0:.1f}-{hi - t0:.1f} s -> {out}')


if __name__ == '__main__':
    main()

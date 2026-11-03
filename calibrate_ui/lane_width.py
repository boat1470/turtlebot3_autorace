#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Measure the lane's width in the projected image, for lane.half_lane_px.

    python3 calibrate_ui/lane_width.py            3 s of frames
    python3 calibrate_ui/lane_width.py 6          6 s
    python3 calibrate_ui/lane_width.py 3 /camera/image_projected

Stand the robot still in the middle of a straight, square to it, with the
yellow line on the left and the white line on the right in view, and the
camera running. Read-only: it subscribes and prints, nothing else.

By default it reads /camera/image_projected, the raw Image that
image_projection (the launcher's extrinsic step) publishes and the very topic
detect_lane takes: arx_mission/launch/detect_lane.launch.py remaps
/detect/image_input onto it and detect_lane subscribes raw. detect_lane need
not be up. Another topic, raw or .../compressed, can be given instead.

half_lane_px is how far detect_lane puts the centre from the one line it can
see; with both in view it averages them instead. If the number is not half the
lane's real width in /camera/image_projected, the centre jumps by the difference
every time a line is lost or found, and the D term turns the step into a kick
(lane.yaml has the arithmetic). It is one measurement with projection.yaml, so
take it again after every extrinsic calibration - and after anything that
moves the camera, like bigger wheels.

The thresholds, lookahead_row and the current half_lane_px are read from
arx_mission/param/lane.yaml, so this sees the lines the way detect_lane does
as long as the values tuned in rqt have been saved there.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import yaml

TOPIC = '/camera/image_projected'

LANE_YAML = Path(__file__).resolve().parent.parent / 'arx_mission' / 'param' / 'lane.yaml'
BAND = 4            # rows either side, averaged


def load_lane_yaml():
    p = yaml.safe_load(LANE_YAML.read_text())['/**']['ros__parameters']

    def hsv(c):
        return ((c['hue_l'], c['saturation_l'], c['lightness_l']),
                (c['hue_h'], c['saturation_h'], c['lightness_h']))
    return (hsv(p['detect']['lane']['white']), hsv(p['detect']['lane']['yellow']),
            p['lane']['lookahead_row'], p['lane']['half_lane_px'])


def line_x(mask_row, side):
    """Middle of the longest run of set pixels on one side of the frame."""
    w = mask_row.size
    lo, hi = (0, w // 2 + w // 8) if side == 'left' else (w // 2 - w // 8, w)
    xs = np.flatnonzero(mask_row[lo:hi]) + lo
    if xs.size < 3:
        return None
    best = max(np.split(xs, np.flatnonzero(np.diff(xs) > 2) + 1), key=len)
    return float(best.mean()) if len(best) >= 3 else None


def to_bgr(msg):
    if isinstance(msg, CompressedImage):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if msg.encoding not in ('bgr8', 'rgb8'):
        return None
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img


class LaneWidth(Node):
    def __init__(self, white, yellow, rows):
        super().__init__('lane_width')
        self.white, self.yellow, self.rows = white, yellow, rows
        self.samples = {r: [] for r in rows}
        self.frames = 0
        self.skipped = None

    def listen(self, topic):
        kind = CompressedImage if topic.endswith('/compressed') else Image
        self.create_subscription(kind, topic, self.cb, 1)

    def cb(self, msg):
        img = to_bgr(msg)
        if img is None:
            self.skipped = getattr(msg, 'encoding', 'undecodable')
            return
        self.frames += 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, *self.white)
        yellow = cv2.inRange(hsv, *self.yellow)
        for r in self.rows:
            if r >= img.shape[0]:
                continue
            band = slice(max(r - BAND, 0), r + BAND + 1)
            y = line_x(yellow[band].max(axis=0), 'left')
            w = line_x(white[band].max(axis=0), 'right')
            if y is not None and w is not None and w > y:
                self.samples[r].append((y, w))


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    white, yellow, look, half_now = load_lane_yaml()
    rows = sorted({300, 350, 480, look})
    rclpy.init()
    node = LaneWidth(white, yellow, rows)
    topic = sys.argv[2] if len(sys.argv) > 2 else TOPIC
    node.listen(topic)
    print(f'reading {topic}')
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f'{node.frames} frames in {secs:g} s')
    if node.skipped:
        print(f'frames skipped: encoding {node.skipped} (bgr8 / rgb8 / compressed only)')
    if not node.frames:
        print(f'nothing on {topic} - camera up, ROS_DOMAIN_ID 30?')
    for r in rows:
        s = np.array(node.samples[r])
        if len(s) < max(3, node.frames // 2):
            print(f'row {r}: both lines in only {len(s)} of {node.frames} frames - skipped')
            continue
        width = s[:, 1] - s[:, 0]
        mark = '  <- lookahead_row' if r == look else ''
        print(f'row {r}: yellow {np.median(s[:, 0]):6.1f}  white {np.median(s[:, 1]):6.1f}  '
              f'width {np.median(width):6.1f} +-{np.std(width):.1f}  half {np.median(width) / 2:6.1f}  '
              f'centre {np.median(s.mean(axis=1)):6.1f}{mark}')
    print(f'half_lane_px now {half_now} (lane.yaml); set it to the half at row {look}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

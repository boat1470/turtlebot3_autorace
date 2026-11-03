#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Measure /odom against a tape, for wheels.radius and wheels.separation.

    python3 calibrate_ui/odom_cal.py straight 2.000     drive 2.000 m by tape
    python3 calibrate_ui/odom_cal.py turn 5             turn 5 full turns

Read-only: it subscribes to /odom and prints, nothing else - drive the robot
with teleop (slowly, 0.05-0.10 m/s) or push it by hand with the motors off.
Enter starts the count, Enter again ends it.

straight: odom's distance against the real one gives a new wheels.radius in
the Pi's turtlebot3_bringup/param/burger.yaml:

    new radius = old radius x real / odom

turn: odom's yaw summed over the turns against 360 x turns. With
use_imu true the yaw is the IMU's and wheels.separation does not enter it;
the number then says how far the IMU drifts. With use_imu false:

    new separation = old separation x odom / real

Do each three times and take the middle one. Mark the floor at the centre of
the axle, not at the caster, and line the robot up square to the tape.
"""

import math
import sys
import threading

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class OdomCal(Node):

    def __init__(self):
        super().__init__('odom_cal')
        self.last = None
        self.dist = 0.0
        self.yaw = 0.0
        self.counting = False
        self.start = None
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)

    def on_odom(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        here = (p.position.x, p.position.y, yaw)
        if self.counting and self.last is not None:
            self.dist += math.hypot(here[0] - self.last[0], here[1] - self.last[1])
            # Wrapped step, so a turn through +-180 sums instead of jumping.
            self.yaw += (here[2] - self.last[2] + math.pi) % (2 * math.pi) - math.pi
        self.last = here


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ('straight', 'turn'):
        print(__doc__)
        return
    mode, real = sys.argv[1], float(sys.argv[2])
    rclpy.init()
    node = OdomCal()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin)
    spinner.start()
    try:
        measure(node, mode, real)
    finally:
        # Stopped before shutdown, or the C++ side aborts on the way out.
        executor.shutdown()
        spinner.join()
        node.destroy_node()
        rclpy.try_shutdown()


def measure(node, mode, real):

    input('Robot on the start mark. Enter to start counting...')
    if node.last is None:
        print('no /odom yet - is the bringup running, on the same ROS_DOMAIN_ID?')
        return
    node.dist, node.yaw, node.counting = 0.0, 0.0, True
    node.start = node.last
    input('Now drive. Enter when the robot is on the end mark...')
    node.counting = False

    if mode == 'straight':
        # Start to end in a line, which is what the tape measures. The summed
        # path is longer by any weaving on the way, so it is shown, not used.
        line = math.hypot(node.last[0] - node.start[0], node.last[1] - node.start[1])
        ratio = real / line if line > 0 else float('nan')
        print(f'odom start->end {line:.4f} m (path {node.dist:.4f} m), tape {real:.4f} m, '
              f'real/odom {ratio:.4f}')
        print(f'  new wheels.radius = old x {ratio:.4f}   (0.042 -> {0.042 * ratio:.5f})')
    else:
        want = 360.0 * real
        got = abs(math.degrees(node.yaw))
        print(f'odom {got:.1f} deg, real {want:.1f} deg, odom/real {got / want:.4f}')
        print(f'  use_imu false: new wheels.separation = old x {got / want:.4f}   '
              f'(0.160 -> {0.160 * got / want:.5f})')
        print('  use_imu true: this is the IMU - separation does not change it')


if __name__ == '__main__':
    main()

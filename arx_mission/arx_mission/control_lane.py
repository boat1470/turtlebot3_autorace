#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
#
# Forked into arx_mission by boat1470, 2026. Same filename and structure as
# the original so the two can be diffed directly; every change is marked ARX.
# The PD control below the gate is untouched - once driving is enabled this
# behaves exactly like the original.
#
# The change is that driving is now off until something says otherwise. The
# original starts with avoid_active False and MAX_VEL 0.1, so it drives the
# instant the first /detect/lane arrives, and the only way to hold it is to
# get /avoid_active true there first. That is a race: DDS discovery between
# two freshly started nodes was measured at 1.49 s here, which at 0.05 m/s is
# about 7.5 cm of travel - and the rules fail the traffic light mission
# outright for crossing the start line on anything but green.
#
# Defaulting to stopped removes the race rather than narrowing it, and it
# leaves /avoid_active and /avoid_control free for avoid_construction to use
# for what they were named for.

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float64


class ControlLane(Node):

    def __init__(self):
        super().__init__('control_lane')

        self.sub_lane = self.create_subscription(
            Float64,
            '/control/lane',
            self.callback_follow_lane,
            1
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.callback_get_max_vel,
            1
        )
        self.sub_avoid_cmd = self.create_subscription(
            Twist,
            '/avoid_control',
            self.callback_avoid_cmd,
            1
        )
        self.sub_avoid_active = self.create_subscription(
            Bool,
            '/avoid_active',
            self.callback_avoid_active,
            1
        )
        # ARX: the go signal, published by arx_mission's mission_control.
        self.sub_drive_enable = self.create_subscription(
            Bool,
            '/arx/drive_enable',
            self.callback_drive_enable,
            1
        )

        self.pub_cmd_vel = self.create_publisher(
            TwistStamped,
            '/control/cmd_vel',
            1
        )

        # PD control related variables
        self.last_error = 0
        self.MAX_VEL = 0.1

        # ARX: the original's hard-coded ceiling, now a parameter. Left at the
        # same value so nothing changes until someone chooses to raise it -
        # but it has to be reachable, because min(..., 0.05) makes the lap
        # impossible: the course is 16-18 m, which at 0.05 m/s is 320-360 s
        # against a 300 s limit, before any mission stops are counted.
        self.declare_parameter('speed.max', 0.05)

        # Avoidance mode related variables
        self.avoid_active = False
        self.avoid_twist = Twist()

        # ARX: stopped until told to drive.
        self.driving = False
        # Zero is published on a timer rather than only from the lane
        # callback, so a hold applied while already moving takes effect even
        # if detect_lane goes quiet - otherwise the last non-zero command
        # simply stands.
        self.hold_timer = self.create_timer(0.1, self.hold_tick)

    def callback_get_max_vel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    def callback_follow_lane(self, desired_center):
        """
        Receive lane center data to generate lane following control commands.

        If avoidance mode is enabled, lane following control is ignored.
        """
        if not self.driving:      # ARX
            return
        if self.avoid_active:
            return

        center = desired_center.data
        error = center - 500

        Kp = 0.0025
        Kd = 0.007

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        twist = Twist()
        # Linear velocity: adjust speed based on error (maximum 0.05 limit)
        ceiling = self.get_parameter('speed.max').value      # ARX: was a literal 0.05
        twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), ceiling)
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        self.publish_cmd_vel(twist)

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg

        if self.avoid_active:
            self.publish_cmd_vel(self.avoid_twist)

    def callback_avoid_active(self, bool_msg):
        self.avoid_active = bool_msg.data
        if self.avoid_active:
            self.get_logger().info('Avoidance mode activated.')
        else:
            self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')

    # ARX: added.
    def callback_drive_enable(self, bool_msg):
        if bool_msg.data == self.driving:
            return
        self.driving = bool_msg.data
        # The D term is a difference against the previous sample, so resuming
        # with an error from before the stop produces a one-frame kick. The
        # original has the same flaw across an avoidance episode.
        self.last_error = 0
        self.get_logger().info(
            'driving enabled' if self.driving else 'driving disabled - holding')

    # ARX: added.
    def hold_tick(self):
        if not self.driving:
            self.publish_cmd_vel(Twist())

    def publish_cmd_vel(self, twist):
        """Stamp a Twist and publish it as the TwistStamped the bridge expects."""
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = self.get_clock().now().to_msg()
        cmd_vel.header.frame_id = ''
        cmd_vel.twist = twist
        self.pub_cmd_vel.publish(cmd_vel)

    def shut_down(self):
        self.get_logger().info('Shutting down. cmd_vel will be 0')
        self.publish_cmd_vel(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

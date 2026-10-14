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
from std_msgs.msg import UInt8


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
        # ARX: detect_lane changes how it computes the centre when it
        # switches between seeing both lines and seeing one - mean of the two
        # versus line plus or minus half a lane width. That moves the target
        # 150 px in a single frame while the robot has not moved at all, and
        # the derivative reads it as enormous speed. Two of the four worst
        # kicks in a traced run were exactly this.
        self.sub_lane_state = self.create_subscription(
            UInt8,
            '/detect/lane_state',
            self.callback_lane_state,
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

        # ARX: the gains were literals, and there was no way to retune
        # them without editing and rebuilding.
        self.declare_parameter('control.kp', 0.0025)
        # A rate now, not a raw frame-to-frame difference. Measured on this
        # course, /detect/lane arrives at 1.9 Hz - frames where the lane is
        # lost publish nothing at all - so `error - last_error` was a step
        # taken over 0.53 s. Multiplied by the original 0.007 that reached
        # 2 rad/s and saturated the clamp: the D term was whipping the robot
        # rather than damping it, and it outweighed the P term in 45 of 146
        # updates. Dividing by the measured dt makes the gain mean the same
        # thing at any frame rate.
        #
        # 0.0037 is 0.007 x 0.53, so the behaviour at today's rate is
        # unchanged - and stays put when lane.frame_skip changes.
        self.declare_parameter('control.kd', 0.0037)
        # Most the published angular velocity may change between one lane
        # message and the next. The backstop for smoothness whatever the
        # error does; 0 turns it off.
        self.declare_parameter('control.max_angular_step', 0.35)
        # ARX: scale kp and max_angular_step with the speed being driven.
        # kp alone gives omega = kp x error whatever the speed, and the radius
        # the robot actually turns on is v / omega - so a gain tuned at one
        # speed runs wide at any higher one, which is what happened at 17B
        # and the U-turns once max_vel went up. Scaling kp by v / speed_ref
        # keeps the radius for a given error the same at every speed.
        # speed_ref is the speed the gains were tuned at; 0 turns this off.
        # kd is not scaled: what bounds it is measurement noise, not speed.
        self.declare_parameter('control.speed_ref', 0.0)
        # ARX: take the speed term's error from /detect/lane_ahead, a second
        # centre detect_lane reads further up the frame (lane.speed_row),
        # when it is fresher than this. The near row that steers is the one
        # that stopped the robot cutting corners, but it also sees a curve
        # late, so braking from it alone is late. The larger of the two
        # errors sets the speed; the near one alone still steers.
        self.declare_parameter('control.speed_from_ahead', True)
        # ARX: how hard the error cuts the speed: max_vel x (1 - e/500) ^ this.
        # The original's 2.2 has 100 px of error already down to 0.61 x, which
        # with the gains tuned at 0.07 m/s was what kept the robot inside the
        # curves. With speed_ref scaling the gain instead, the cut can be
        # gentler: 1.0 is linear, 1.5 keeps 0.77 x at 100 px.
        self.declare_parameter('control.slow_power', 2.2)
        # ARX: what one wheel can do, and half the track. In a turn the outer
        # wheel runs at v + |omega| x half_track, and the Burger's wheels top
        # out at 0.22 m/s: asked for more, the wheel saturates and the turn
        # comes out wider than commanded - into the outer line. So v is cut
        # to keep the outer wheel inside wheel_max, and the straights can run
        # at the wheel limit without the curves paying for it. 0 turns it off.
        # 0.21: a linear.x of 0.22 reaches the OpenCR as 22 cm/s and is
        # refused outright - the robot does not move (2026-10-13).
        self.declare_parameter('control.wheel_max', 0.21)
        self.declare_parameter('control.half_track', 0.08)
        self.declare_parameter('control.ahead_max_age_s', 0.5)
        self.ahead_error = None
        self.ahead_time = None
        self.sub_lane_ahead = self.create_subscription(
            Float64, '/detect/lane_ahead', self.callback_lane_ahead, 1)

        # PD control related variables
        # ARX: None rather than 0 - "no previous sample", so the first
        # update after a start, a stop or a lane_state change contributes no
        # derivative at all instead of a spurious one.
        self.last_error = None
        self.last_time = None
        self.last_angular = None
        self.lane_state = None

        # ARX: was a hard-coded 0.1, and it is the ceiling that actually
        # bites first. linear.x is
        #
        #     min(max_vel * (1 - |error|/500) ** 2.2, speed.max)
        #
        # so with max_vel at 0.1 the speed cannot pass 0.1 however high
        # speed.max is set, and the error term holds it below that most of
        # the time - at an error of 100 px it is already down to 0.06. An
        # invisible ceiling is the wrong thing to have in the way while the
        # lap time is the problem.
        #
        # /control/max_vel still overrides it at run time; that is how the
        # level crossing mission is meant to slow the robot down. This is the
        # value it starts from.
        self.declare_parameter('control.max_vel', 0.1)
        self.MAX_VEL = self.get_parameter('control.max_vel').value

        # ARX: the original's hard-coded ceiling, now a parameter. Left at the
        # same value so nothing changes until someone chooses to raise it -
        # but it has to be reachable, because min(..., 0.05) makes the lap
        # impossible: the course is 16-18 m, which at 0.05 m/s is 320-360 s
        # against a 300 s limit, before any mission stops are counted.
        self.declare_parameter('speed.max', 0.05)

        ref = self.get_parameter('control.speed_ref').value
        if ref > 0:
            self.get_logger().info(
                f'kp and max_angular_step scale with speed, 1.0 at {ref:.3f} m/s')

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
        # ARX: log it. This overrides control.max_vel for the rest of the
        # run and nothing said so before, which would make a robot that had
        # quietly been slowed impossible to tell from one that was badly
        # tuned.
        if max_vel_msg.data != self.MAX_VEL:
            self.get_logger().info(
                f'max_vel overridden on /control/max_vel: '
                f'{self.MAX_VEL:.3f} -> {max_vel_msg.data:.3f} m/s')
        self.MAX_VEL = max_vel_msg.data

    def callback_lane_ahead(self, msg):
        """The lane centre further up the frame, for braking only (ARX)."""
        self.ahead_error = msg.data - 500
        self.ahead_time = self.get_clock().now().nanoseconds * 1e-9

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

        # ARX: a real derivative, over the time that actually elapsed.
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = None if self.last_time is None else now - self.last_time
        self.last_time = now

        derivative = 0.0
        if self.last_error is not None and dt is not None and dt > 1e-3:
            derivative = (error - self.last_error) / dt
        self.last_error = error

        # ARX: the speed first, because the steering gain scales with it.
        # Its error is the larger of the near row's and the ahead row's, so a
        # curve that is still ahead of the steering row already slows the
        # robot down.
        speed_error = abs(error)
        # Not a centre off the picture, though: a short fit stretched to the
        # ahead row can land thousands of pixels out - +4434 at the end of
        # the lot's yellow walls in run 1014_2251, which cut v to 0 and left
        # it there, the next frames having no lane to replace it.
        if (self.get_parameter('control.speed_from_ahead').value
                and self.ahead_time is not None
                and now - self.ahead_time <= self.get_parameter('control.ahead_max_age_s').value
                and abs(self.ahead_error) <= 500):
            speed_error = max(speed_error, abs(self.ahead_error))

        twist = Twist()
        # Linear velocity: adjust speed based on error (maximum 0.05 limit)
        ceiling = self.get_parameter('speed.max').value      # ARX: was a literal 0.05
        twist.linear.x = min(
            self.MAX_VEL * (max(1 - speed_error / 500, 0)
                            ** self.get_parameter('control.slow_power').value), ceiling)

        # ARX: see control.speed_ref and control.wheel_max. The gain scales
        # with the speed, and the speed is cut for the wheels once omega is
        # known - so it goes round twice, the second pass with the cut speed,
        # and the pair that goes out is consistent. Floored at 0.04 m/s so a
        # robot slowed right down by a big error can still steer out of it.
        ref = self.get_parameter('control.speed_ref').value
        kp = self.get_parameter('control.kp').value
        kd = self.get_parameter('control.kd').value
        step = self.get_parameter('control.max_angular_step').value
        wheel_max = self.get_parameter('control.wheel_max').value
        half_track = self.get_parameter('control.half_track').value
        linear = twist.linear.x
        for _ in range(2):
            scale = max(linear, 0.04) / ref if ref > 0 else 1.0
            angular_z = kp * scale * error + kd * derivative
            angular = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
            # ARX: rate limit. Whatever the error does, the wheels are not
            # asked to change what they are doing faster than this.
            if step > 0 and self.last_angular is not None:
                angular = max(min(angular, self.last_angular + step * scale),
                              self.last_angular - step * scale)
            if wheel_max <= 0:
                break
            linear = max(0.0, min(twist.linear.x, wheel_max - abs(angular) * half_track))
        self.last_angular = angular
        twist.angular.z = angular
        twist.linear.x = linear
        self.publish_cmd_vel(twist)

    # ARX: added.
    def callback_lane_state(self, msg):
        """Forget the previous error when the centre changes meaning."""
        if msg.data == self.lane_state:
            return
        self.lane_state = msg.data
        self.last_error = None

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg

        if self.avoid_active:
            self.publish_cmd_vel(self.avoid_twist)

    def callback_avoid_active(self, bool_msg):
        # ARX: log on the change, not on every message. The original logged
        # each one, and mission_control republishes this every tick so a node
        # restarted mid-run picks the state up - which is ten identical lines
        # a second, drowning everything else in the log. Same fix as
        # callback_drive_enable below, which this file already had.
        if bool_msg.data == self.avoid_active:
            return
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
        self.last_error = None
        self.last_time = None
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

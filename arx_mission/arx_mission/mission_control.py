#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Mission sequencer: decides who has the wheel, and when.

The ROBOTIS example has no such node - the ROS 1 core_node_mission was never
ported - so each mission runs on its own with nothing chaining them. This is
that missing piece. It starts with the traffic light and grows one mission at
a time.

Holding the robot is done by /arx/drive_enable, which the forked control_lane
in this package starts out with as false. Two earlier attempts were worse.

Publishing 0.0 on /control/max_vel zeroes control_lane's linear.x but not its
angular.z, which is not gated by max_vel at all - the robot stands there
spinning at up to 2 rad/s and is pointing the wrong way by the time the light
changes.

Driving a zero Twist through /avoid_active and /avoid_control does stop it,
but only once those messages have arrived: DDS discovery between two freshly
started nodes was measured at 1.49 s here, and the original control_lane
drives from its first lane message. That is a race, and narrowing it is not
the same as removing it. It also occupies the two topics avoid_construction
needs for what they were named for.

Defaulting control_lane to stopped inverts it: nothing can make the robot
move until this node says so.

The timeout is not a nicety either. The official clock starts at the first
green whether or not the robot moves, so waiting costs nothing but never
starting costs the entire run, not just this mission. Failing to see green has
to end in driving off anyway.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, UInt8

# /arx/traffic_light - must match detect_traffic_light.py
LIGHT_UNKNOWN = 0
LIGHT_RED = 1
LIGHT_YELLOW = 2
LIGHT_GREEN = 3

NAMES = {LIGHT_UNKNOWN: 'unknown', LIGHT_RED: 'red',
         LIGHT_YELLOW: 'yellow', LIGHT_GREEN: 'green'}

# /arx/armed_mission - tells a detector to do its work, and every other
# detector to idle. On the competition Raspberry Pi they cannot all run at
# once, and nothing downstream reads a detector that is not armed anyway.
MISSION_NONE = 0
MISSION_TRAFFIC_LIGHT = 1

STAGE_WAIT_GREEN = 'wait_green'
STAGE_DRIVING = 'driving'


class MissionControl(Node):

    def __init__(self):
        super().__init__('arx_mission_control')

        self.declare_parameter('light.enabled', True)
        self.declare_parameter('light.confirm_frames', 3)
        self.declare_parameter('light.give_up_s', 25.0)
        # Require the light to have been seen NOT green before a green is
        # acted on, so what releases the robot is the change to green rather
        # than a green that was already burning when the stack came up.
        #
        # It matters because green lasts five seconds and the rules fail the
        # mission for crossing the start line on any other signal. Joining a
        # green with a second left to run means confirming for 0.4 s, then
        # accelerating from rest, then reaching the line - by which time it is
        # red. In simulation the light cycles from t=0 regardless, so roughly
        # five elevenths of all starts land mid-green.
        #
        # On a real course the cycle is expected to begin at red, so this
        # costs nothing there; the parameter exists in case practice day says
        # otherwise.
        self.declare_parameter('light.require_transition', True)
        self.declare_parameter('rate_hz', 10.0)

        self.stage = STAGE_WAIT_GREEN
        self.stage_since = None
        self.light = LIGHT_UNKNOWN
        self.green_frames = 0
        self.not_green_frames = 0
        self.seen_not_green = False

        self.pub_armed = self.create_publisher(UInt8, '/arx/armed_mission', 1)
        # Republished every tick. Idempotent, and it means a control_lane
        # restarted mid-run picks the state up within one tick instead of
        # sitting held forever.
        self.pub_drive = self.create_publisher(Bool, '/arx/drive_enable', 1)
        self.create_subscription(UInt8, '/arx/traffic_light', self.on_light, 1)

        self.timer = self.create_timer(
            1.0 / max(self.get_parameter('rate_hz').value, 0.1), self.tick)
        self.get_logger().info(f'stage -> {self.stage}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_light(self, msg):
        self.light = msg.data
        # Counted per message rather than per tick so the confirmation is in
        # camera frames, not in ticks of a free-running timer.
        if msg.data == LIGHT_GREEN:
            self.green_frames += 1
            self.not_green_frames = 0
            return

        self.green_frames = 0
        self.not_green_frames += 1
        # LIGHT_UNKNOWN counts towards this, not just red and amber. The
        # shipped red band is hue 0-24, which misses the wrap-around at
        # 160-179 where a real lamp may well sit, so a red phase can easily
        # read as "nothing seen" - and waiting for a red that never arrives
        # would mean sitting out the give-up timer on every run.
        #
        # Requiring several consecutive frames is what makes that safe: one
        # dropped detection in the middle of a green phase must not satisfy
        # the condition, or this guard buys nothing.
        need = self.get_parameter('light.confirm_frames').value
        if not self.seen_not_green and self.not_green_frames >= need:
            self.seen_not_green = True
            self.get_logger().info(
                f'light was not green for {self.not_green_frames} frames '
                f'({NAMES.get(msg.data, msg.data)}) - a green may now be acted on')

    def go(self, stage, why):
        self.get_logger().info(f'stage {self.stage} -> {stage}: {why}')
        self.stage = stage
        self.stage_since = self.now()

    def set_driving(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.pub_drive.publish(msg)

    def tick(self):
        now = self.now()
        if self.stage_since is None:
            self.stage_since = now

        if self.stage == STAGE_WAIT_GREEN:
            self.set_armed(MISSION_TRAFFIC_LIGHT)
            self.set_driving(False)
            if not self.get_parameter('light.enabled').value:
                self.start('traffic light check disabled')
                return
            need = self.get_parameter('light.confirm_frames').value
            ready = (self.seen_not_green
                     or not self.get_parameter('light.require_transition').value)
            if ready and self.green_frames >= need:
                self.start(f'green for {self.green_frames} frames')
                return
            waited = now - self.stage_since
            if waited > self.get_parameter('light.give_up_s').value:
                # Never sit here past the give-up time. The clock has been
                # running since the light first went green, so a light we
                # cannot see costs less than the rest of the course.
                #
                # The reason is logged because the two causes need different
                # fixes: never seeing green at all points at the green HSV
                # band or the region of interest, while seeing green but
                # never a non-green phase means require_transition is holding
                # on a light that was already green when we arrived.
                why = (f'last seen {NAMES.get(self.light, self.light)}, '
                       f'green streak {self.green_frames}, '
                       f'non-green phase seen: {self.seen_not_green}')
                self.get_logger().warn(
                    f'no usable green after {waited:.0f} s ({why}) - driving anyway')
                self.start('gave up waiting')
            return

        # STAGE_DRIVING. The light is deliberately not looked at again: past
        # the junction there are other round coloured things beside the road,
        # and stopping for one of them in the middle of the course would be
        # worse than anything the light itself can cost.
        self.set_armed(MISSION_NONE)
        self.set_driving(True)

    def start(self, why):
        self.go(STAGE_DRIVING, why)
        self.set_driving(True)

    def set_armed(self, mission):
        msg = UInt8()
        msg.data = int(mission)
        self.pub_armed.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

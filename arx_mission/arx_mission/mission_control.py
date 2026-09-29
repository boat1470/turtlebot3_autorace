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

Mission two, the intersection, is the same shape: the forked
detect_intersection_sign reports what it sees in each frame with no memory of
its own, and all the hysteresis lives here. For now identifying the arrow ends
in a stop, because there is nothing to hand over to yet - the turn itself is
the next piece of work, not this one.
"""

from collections import deque
import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
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

# /detect/traffic_sign - must match detect_intersection_sign.py. The values
# come from the example's Enum('TrafficSign', 'intersection left right') and
# are kept even though the fork no longer matches the intersection sign
# itself, so the numbers on the wire still mean what the example's tooling
# thinks they mean. 1 is therefore not produced by our detector.
SIGN_CONSTRUCTION = 1
SIGN_LEFT = 2
SIGN_RIGHT = 3

SIGN_NAMES = {SIGN_CONSTRUCTION: 'construction',
              SIGN_LEFT: 'left', SIGN_RIGHT: 'right'}

# /arx/follow_side - must match detect_lane.py. Which lane line detect_lane
# should steer by once the junction has been decided: the yellow one on the
# left, the white one on the right, or the usual mean of the two.
FOLLOW_AUTO = 0
FOLLOW_YELLOW = 1
FOLLOW_WHITE = 2

SIDE_FOR_SIGN = {SIGN_LEFT: FOLLOW_YELLOW, SIGN_RIGHT: FOLLOW_WHITE}
SIDE_NAMES = {FOLLOW_AUTO: 'auto', FOLLOW_YELLOW: 'yellow', FOLLOW_WHITE: 'white'}

# /arx/armed_mission - tells a detector to do its work, and every other
# detector to idle. On the competition Raspberry Pi they cannot all run at
# once, and nothing downstream reads a detector that is not armed anyway.
MISSION_NONE = 0
MISSION_TRAFFIC_LIGHT = 1
MISSION_INTERSECTION = 2
MISSION_CONSTRUCTION = 3

STAGE_WAIT_GREEN = 'wait_green'
STAGE_DRIVE_TO_SIGN = 'drive_to_sign'
STAGE_TURN = 'turn'
STAGE_FOLLOW_SIDE = 'follow_side'
STAGE_DRIVE_TO_CONSTRUCTION = 'drive_to_construction'
STAGE_AVOID_CONSTRUCTION = 'avoid_construction'

# /detect/lane_state, from detect_lane: which line it is steering by.
LANE_STATE_FOR_SIDE = {FOLLOW_YELLOW: 1, FOLLOW_WHITE: 3}
STAGE_STOPPED = 'stopped'

# Every stage `start_stage` will accept (ARX). Written out rather than
# collected from the constants above so that adding a stage without deciding
# whether a run may start in it is a deliberate act.
STAGES = frozenset({
    STAGE_WAIT_GREEN,
    STAGE_DRIVE_TO_SIGN,
    STAGE_TURN,
    STAGE_FOLLOW_SIDE,
    STAGE_DRIVE_TO_CONSTRUCTION,
    STAGE_AVOID_CONSTRUCTION,
    STAGE_STOPPED,
})

# /detect/obstacle, from detect_obstacle: which of the three strips ahead of
# the robot has a board in it. Relative to the robot, not to the corridor -
# a robot hugging the white line is already in the right half, so a board
# blocking that half is straight in front of it.
BLOCKED_NONE = 0
BLOCKED_AHEAD = 1
BLOCKED_LEFT = 2
BLOCKED_RIGHT = 4


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

        self.declare_parameter('intersection.enabled', True)
        # A majority of the last `window` reports, not a run of identical ones.
        #
        # The traffic light can use a consecutive run because its detector is
        # nearly always right and merely intermittent. The sign detector is
        # not: measured against the left board at five distances, 163 frames
        # produced a verdict and 76% of them said left. The wrong answers are
        # scattered through the right ones rather than clustered, so a run of
        # three identical reports can take a long time to appear - at 0.35 m
        # the reports came out 9 left to 8 right, interleaved - while a
        # majority is decided almost immediately.
        #
        # Requiring a score margin in the detector instead was measured and
        # rejected: even at a margin of 2.5, twenty frames still called a left
        # board right with confidence, and the message rate dropped by 60%.
        # The errors are not weak-confidence errors, so voting is what
        # removes them.
        self.declare_parameter('intersection.window', 7)
        self.declare_parameter('intersection.confirm_votes', 5)
        # How old a report may be and still count.
        #
        # Without this the window is seven reports however far apart they
        # arrived, and the detector goes quiet whenever the board is out of
        # frame. One run put reports fourteen seconds apart in the same
        # window - by then the robot was metres past the junction, looking at
        # something else entirely, and those reports were being weighed
        # against ones taken while the board was in view.
        #
        # Three seconds is a few metres of frame at this speed and comfortably
        # longer than the gaps between reports while the board is actually
        # being seen.
        self.declare_parameter('intersection.vote_max_age_s', 3.0)
        # Stop once the arrow is identified instead of acting on it. Purely
        # a debug aid now that there is something to hand over to.
        self.declare_parameter('intersection.stop_after_detect', False)
        # Measured from the moment the robot is released, not from the sign
        # coming into view, so it has to cover the whole approach.
        self.declare_parameter('intersection.give_up_s', 40.0)

        # Turn towards the chosen line before handing over to it.
        #
        # Measured: at the moment the arrow is confirmed the robot is already
        # past the board and pointing along the course, and the yellow line
        # is a fragment in the top-left of the bird's-eye view - 8802 px,
        # enough to pass the 3000 threshold and be fitted, but spanning only
        # 309 of 600 rows. detect_lane does steer by it, and reports
        # lane_state 1, but the fit sits near the middle of the frame rather
        # than to the left, so `line + 280` puts the target at 714 out of
        # 1000 and control_lane turns *right*. One run went from 158 degrees
        # to 103 - clockwise - and took the wrong branch while reporting
        # lane_state 1 the whole way.
        #
        # Turning towards the line first brings it properly into frame before
        # anything steers by it.
        self.declare_parameter('intersection.turn.enabled', True)
        self.declare_parameter('intersection.turn.rate', 0.4)
        self.declare_parameter('intersection.turn.speed', 0.0)
        # How far to turn once the arrow is known, per side. Positive is
        # counter-clockwise, so a left arrow is positive and a right one
        # negative. The two are separate numbers because the junction is not
        # symmetric.
        #
        # A fixed amount of rotation, not a heading to reach and not "turn
        # until the line looks right". Both of those were tried. Aiming at a
        # heading needs the heading the robot set off with, which is only
        # right while this node has been up since the start. Waiting on the
        # lane detector cannot tell a line that is merely visible from one
        # that is usable: it handed over while the yellow fit still sat in
        # the middle of the frame, `line + 280` put the steering target at
        # 714 of 1000, and the robot turned right into the wrong branch while
        # reporting lane_state 1 throughout.
        # What the arrow is used for once it has been read.
        #
        # False, the default: only to decide which way to turn. The turn
        # itself puts the robot into the branch, and lane following carries
        # on exactly as it does everywhere else - whatever lane the robot now
        # points into is the lane it follows.
        #
        # True also pins detect_lane to that side's line for the rest of the
        # run. Measured on this course, that steers by `line + 280`, and 280
        # is half a lane width - so it aims at the middle of the lane, not at
        # the line. The turn it produced had a 0.41 m radius and needed 0.64 m
        # of travel to come round 90 degrees, which the junction does not
        # give it.
        self.declare_parameter('intersection.follow_chosen_line', False)

        self.declare_parameter('intersection.turn.left_deg', 30.0)
        self.declare_parameter('intersection.turn.right_deg', -5.0)
        # Ease in over the last stretch so a tick of rotation cannot overshoot
        # by much. At 0.4 rad/s and 10 Hz a tick is 2.3 degrees. Capped at
        # half the turn, so a 5 degree turn is not spent entirely crawling.
        self.declare_parameter('intersection.turn.slow_within_deg', 10.0)
        # Only a backstop now that the turn has a definite size: it catches
        # odometry that never arrives, which would otherwise mean turning
        # until the race ended.
        self.declare_parameter('intersection.turn.give_up_s', 20.0)

        # Only believe the sign while the robot is pointing the right way.
        #
        # These are absolute headings, read straight off /odom. Measured on a
        # running stack, /odom's yaw is identical to the world yaw that gz
        # reports - only its *position* counts from the spawn point, and the
        # frame is translated without being rotated. An earlier version of
        # this gate measured the turn since the first odometry message
        # instead, on the assumption that the two frames disagreed by 90
        # degrees; they do not, and that reference had the further problem of
        # changing whenever this node was restarted mid-run.
        #
        # 180 +/- 45 means facing west, give or take. The arrow board is a
        # box 0.12 m wide and 0.025 m thick, turned by -90 degrees in the
        # world file, which leaves its broad faces normal to the world x
        # axis. Looking along y - which is how the robot sets off, and where
        # it ends up after the junction - shows 12 px of its edge, and that
        # has never once matched. Looking along x shows 22 to 93 px of its
        # face, which matched with 27 good and 17 inliers.
        #
        # The window is therefore not about where on the course the robot is.
        # It is about whether the board can be seen at all from where it is
        # pointing, which is why it is worth having: the rules allow signs on
        # the course that belong to no mission, and a report that arrives
        # while the board is edge-on is a report about something else.
        self.declare_parameter('intersection.heading.enabled', True)
        self.declare_parameter('intersection.heading.center_deg', 180.0)
        self.declare_parameter('intersection.heading.tolerance_deg', 45.0)

        # The construction sign, hunted after the junction is behind us.
        # Same shape as the intersection block: the stage reads whichever
        # block sign_prefix() names, so the two cannot be confused.
        self.declare_parameter('construction.enabled', True)
        self.declare_parameter('construction.window', 7)
        self.declare_parameter('construction.confirm_votes', 5)
        self.declare_parameter('construction.vote_max_age_s', 3.0)
        # Stop where the sign was found. The debug step: nothing is built on
        # top of this yet.
        self.declare_parameter('construction.stop_after_detect', True)
        self.declare_parameter('construction.give_up_s', 120.0)
        # The board is turned -90 degrees in the world file like the arrow
        # is, so its broad faces are normal to the world x axis and it can
        # only be read while the robot looks along x. 0 is looking towards
        # +x.
        self.declare_parameter('construction.heading.enabled', True)
        self.declare_parameter('construction.heading.center_deg', 0.0)
        self.declare_parameter('construction.heading.tolerance_deg', 45.0)

        # Threading the boards, once the sign says they are coming.
        #
        # Measured from the course texture, the corridor there is 0.485 m
        # wide - two lane widths - and the boards alternate which half they
        # block. Hugging the white line puts the steering target 6 mm from
        # the middle of the gap the first board leaves, which is why this
        # needs no avoidance controller of its own: detect_lane already
        # knows how to hug a line and control_lane already knows how to
        # drive to it.
        self.declare_parameter('construction.avoid.enabled', True)
        # Stop in front of the first board that blocks the side being
        # hugged, rather than swapping sides and carrying on. The debug step
        # for this leg: swapping is the next piece of work.
        self.declare_parameter('construction.avoid.stop_at_obstacle', True)
        # Never sit in this stage for ever. A detector that says nothing
        # looks exactly like a clear corridor from here.
        self.declare_parameter('construction.avoid.give_up_s', 60.0)

        self.declare_parameter('rate_hz', 10.0)

        # Which stage a run begins in (ARX). The default is the whole course
        # from the start line; anything else skips the missions before it and
        # is for testing one mission without driving to it first, which took
        # minutes per attempt.
        #
        # Only the stage is set. Whatever state the skipped stages would have
        # left behind is not, so a stage started cold has to stand on its own
        # - drive_to_construction does, because `decision` being None makes
        # chosen_side() fall back to the mean of both lines, which is what a
        # run that never read an arrow should steer by anyway.
        #
        # The robot still has to be standing somewhere that stage makes sense
        # from. full.launch.py's `start` argument sets the two together.
        self.declare_parameter('start_stage', STAGE_WAIT_GREEN)

        self.stage = self.get_parameter('start_stage').value
        if self.stage not in STAGES:
            # Loudly, and carry on from the start. An unknown stage reaches
            # tick()'s else branch, which is tick_stopped: the robot would sit
            # there not moving and nothing would say why.
            self.get_logger().error(
                f'start_stage {self.stage!r} is not a stage - '
                f'starting at {STAGE_WAIT_GREEN}. '
                f'Known: {", ".join(sorted(STAGES))}')
            self.stage = STAGE_WAIT_GREEN
        if self.stage != STAGE_WAIT_GREEN:
            self.get_logger().warn(
                f'starting at stage {self.stage} - the missions before it are '
                f'skipped, so this is a test run, not a scoring one')
        self.stage_since = None
        self.light = LIGHT_UNKNOWN
        self.green_frames = 0
        self.not_green_frames = 0
        self.seen_not_green = False
        # Counted so a give-up can say whether the detector was silent or
        # merely unsuccessful. The two look identical from the other
        # counters - no messages at all leaves them at their initial values,
        # which reads the same as a detector that never found anything.
        self.light_msgs = 0

        # (time, value) rather than value alone, so stale reports can be
        # dropped rather than voted on.
        self.sign_votes = deque(maxlen=self.get_parameter('intersection.window').value)
        self.sign_msgs = 0
        # Reports thrown away for pointing the wrong way. Counted separately
        # so a give-up can tell "never saw the sign" from "saw it and refused
        # to look" - the two are indistinguishable from the vote window, and
        # that ambiguity is what made the sign impossible to debug before.
        self.sign_rejected = 0
        self.yaw = None
        # Accumulated, not end minus start. The difference between two
        # headings wraps into -180..180, so a 243 degree turn reads as 117
        # and a cap of 270 could never fire. Caught by a unit test that
        # turned the long way round.
        self.turn_start_yaw = None
        self.turn_last_yaw = None
        self.turn_accum = 0.0
        self.lane_state = None
        # Whether the last report was counted, so the log can say when that
        # changes. Without it the only record of the gate's work was the
        # give-up message, which a successful run never prints - so a run
        # that confirmed left no way of telling how much had been thrown
        # away on the way there.
        self.heading_was_ok = None
        # Which way the course goes, once it has been confirmed. Set once and
        # never revisited - see on_sign.
        self.decision = None
        # Set once the construction sign has been read. From then on the
        # steering goes back to the mean of both lines: the branch the arrow
        # picked is long behind, and there is nothing left to hug a single
        # line for. Not on entering the stage - on reading the sign.
        self.construction_seen = False
        # Latest report from detect_obstacle. None until one arrives, which
        # is not the same as a clear corridor and is treated differently.
        self.blocked = None

        self.pub_armed = self.create_publisher(UInt8, '/arx/armed_mission', 1)
        # Republished every tick. Idempotent, and it means a control_lane
        # restarted mid-run picks the state up within one tick instead of
        # sitting held forever.
        self.pub_drive = self.create_publisher(Bool, '/arx/drive_enable', 1)
        self.pub_side = self.create_publisher(UInt8, '/arx/follow_side', 1)
        # The example's own override path. control_lane returns early from
        # lane following while avoid_active is set and relays whatever
        # arrives on /avoid_control straight to /cmd_vel - without checking
        # its drive_enable gate, so drive_enable must stay true here or
        # hold_tick fights this at 10 Hz.
        self.pub_avoid = self.create_publisher(Bool, '/avoid_active', 1)
        self.pub_avoid_cmd = self.create_publisher(Twist, '/avoid_control', 1)
        self.create_subscription(UInt8, '/detect/lane_state', self.on_lane_state, 1)
        self.create_subscription(UInt8, '/arx/traffic_light', self.on_light, 1)
        self.create_subscription(UInt8, '/detect/traffic_sign', self.on_sign, 1)
        self.create_subscription(Odometry, '/odom', self.on_odom, 1)
        self.create_subscription(UInt8, '/detect/obstacle', self.on_obstacle, 1)

        self.timer = self.create_timer(
            1.0 / max(self.get_parameter('rate_hz').value, 0.1), self.tick)
        self.get_logger().info(f'stage -> {self.stage}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_light(self, msg):
        self.light = msg.data
        self.light_msgs += 1
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

    def on_odom(self, msg):
        """Track the robot's heading, in degrees, -180 to 180."""
        q = msg.pose.pose.orientation
        self.yaw = math.degrees(math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def on_obstacle(self, msg):
        """Which half of the corridor ahead detect_obstacle says is blocked."""
        self.blocked = msg.data

    def heading_ok(self, prefix=None):
        """Whether the robot is pointing a way this mission's board can be seen from."""
        prefix = prefix or self.sign_prefix()
        if not self.get_parameter(f'{prefix}.heading.enabled').value:
            return True
        if self.yaw is None:
            # No odometry. Accepting is the lesser evil: a gate that cannot
            # measure anything and refuses everything would fail the mission
            # without the robot ever being pointed the wrong way.
            self.get_logger().warn(
                'heading gate is on but no /odom has arrived - accepting the sign anyway',
                once=True)
            return True
        want = self.get_parameter(f'{prefix}.heading.center_deg').value
        tol = self.get_parameter(f'{prefix}.heading.tolerance_deg').value
        # Shortest angular distance, so a window straddling +/-180 works like
        # any other.
        return abs((self.yaw - want + 180.0) % 360.0 - 180.0) <= tol

    def on_sign(self, msg):
        """Keep the last `window` reports so they can be voted on."""
        self.sign_msgs += 1
        ok = self.heading_ok()
        if ok != self.heading_was_ok:
            # Only on the change, not every report: this fires a few times a
            # second while the board is in view.
            self.heading_was_ok = ok
            where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            if ok:
                self.get_logger().info(
                    f'facing {where} - sign reports now counted '
                    f'({self.sign_rejected} ignored so far)')
            else:
                want = self.get_parameter('intersection.heading.center_deg').value
                tol = self.get_parameter('intersection.heading.tolerance_deg').value
                self.get_logger().info(
                    f'facing {where} - sign reports ignored, '
                    f'want {want:+.0f}+/-{tol:.0f}')
        if not ok:
            self.sign_rejected += 1
            return
        self.sign_votes.append((self.now(), msg.data))

    def sign_prefix(self):
        """Which parameter block governs the sign hunt in this stage."""
        return ('construction' if self.stage == STAGE_DRIVE_TO_CONSTRUCTION
                else 'intersection')

    def reset_votes(self):
        """Start a fresh hunt: the last mission's reports are not evidence."""
        self.sign_votes.clear()
        self.sign_msgs = 0
        self.sign_rejected = 0
        self.heading_was_ok = None

    def fresh_votes(self):
        """The reports still young enough to count."""
        cutoff = self.now() - self.get_parameter(
            f'{self.sign_prefix()}.vote_max_age_s').value
        return [v for t, v in self.sign_votes if t >= cutoff]

    def sign_majority(self, wanted=None):
        """The value with enough votes in the window, or (None, count).

        `wanted` is the set of numbers this stage is listening for. Only one
        detector is armed at a time and the numbers do not overlap, so one
        vote buffer serves every mission - but a stage should not be able to
        act on a report meant for a different one.
        """
        wanted = wanted if wanted is not None else set(SIGN_NAMES)
        need = self.get_parameter(f'{self.sign_prefix()}.confirm_votes').value
        votes = self.fresh_votes()
        best, count = None, 0
        for value in set(votes):
            c = votes.count(value)
            if c > count:
                best, count = value, c
        if best in wanted and count >= need:
            return best, count
        return None, count

    def go(self, stage, why):
        self.get_logger().info(f'stage {self.stage} -> {stage}: {why}')
        self.stage = stage
        self.stage_since = self.now()

    def set_driving(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.pub_drive.publish(msg)

    def set_armed(self, mission):
        msg = UInt8()
        msg.data = int(mission)
        self.pub_armed.publish(msg)

    def on_lane_state(self, msg):
        """Remember what detect_lane is steering by, for the logs."""
        self.lane_state = msg.data

    def set_avoid(self, active, angular=0.0, linear=0.0):
        """Take the wheel directly, or hand it back to lane following."""
        flag = Bool()
        flag.data = bool(active)
        self.pub_avoid.publish(flag)
        if active:
            # Republished every tick: control_lane forwards a command only
            # when one arrives, on a volatile depth-1 subscription.
            twist = Twist()
            twist.linear.x = float(linear)
            twist.angular.z = float(angular)
            self.pub_avoid_cmd.publish(twist)

    def set_follow_side(self, side):
        msg = UInt8()
        msg.data = int(side)
        self.pub_side.publish(msg)

    def start(self, why):
        self.go(STAGE_DRIVE_TO_SIGN, why)
        self.set_driving(True)

    def stop(self, why):
        self.go(STAGE_STOPPED, why)
        self.set_driving(False)

    def tick(self):
        now = self.now()
        if self.stage_since is None:
            self.stage_since = now

        if self.stage == STAGE_WAIT_GREEN:
            self.tick_wait_green(now)
        elif self.stage == STAGE_DRIVE_TO_SIGN:
            self.tick_drive_to_sign(now)
        elif self.stage == STAGE_TURN:
            self.tick_turn(now)
        elif self.stage == STAGE_FOLLOW_SIDE:
            self.tick_follow_side()
        elif self.stage == STAGE_DRIVE_TO_CONSTRUCTION:
            self.tick_drive_to_construction(now)
        elif self.stage == STAGE_AVOID_CONSTRUCTION:
            self.tick_avoid_construction(now)
        else:
            self.tick_stopped()

    def tick_wait_green(self, now):
        self.set_armed(MISSION_TRAFFIC_LIGHT)
        self.set_driving(False)
        self.set_follow_side(FOLLOW_AUTO)
        self.set_avoid(False)
        if not self.get_parameter('light.enabled').value:
            self.start('traffic light check disabled')
            return
        need = self.get_parameter('light.confirm_frames').value
        ready = (self.seen_not_green or
                 not self.get_parameter('light.require_transition').value)
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
            why = (f'{self.light_msgs} light messages, '
                   f'last seen {NAMES.get(self.light, self.light)}, '
                   f'green streak {self.green_frames}, '
                   f'non-green phase seen: {self.seen_not_green}')
            self.get_logger().warn(
                f'no usable green after {waited:.0f} s ({why}) - driving anyway')
            self.start('gave up waiting')

    def tick_drive_to_sign(self, now):
        # The light is deliberately not looked at again: past the junction
        # there are other round coloured things beside the road, and stopping
        # for one of them in the middle of the course would be worse than
        # anything the light itself can cost.
        self.set_driving(True)
        self.set_follow_side(FOLLOW_AUTO)
        self.set_avoid(False)

        if not self.get_parameter('intersection.enabled').value:
            self.set_armed(MISSION_NONE)
            return

        self.set_armed(MISSION_INTERSECTION)

        winner, votes = self.sign_majority()
        if winner is not None:
            self.decision = winner
            name = SIGN_NAMES[winner]
            self.get_logger().info(
                f'sign {name} confirmed: {votes} of the last '
                f'{len(self.fresh_votes())} reports '
                f'({self.sign_msgs} reports seen, {self.sign_rejected} '
                f'ignored on heading)')
            if self.get_parameter('intersection.stop_after_detect').value:
                self.stop(f'sign {name} - stopping here, as asked')
            elif self.get_parameter('intersection.turn.enabled').value:
                self.turn_start_yaw = self.yaw
                self.turn_last_yaw = self.yaw
                self.turn_accum = 0.0
                self.go(STAGE_TURN,
                        f'sign {name} - turning {self.turn_target():+.0f} deg')
            else:
                self.after_junction(
                    f'sign {name} - steering by the '
                    f'{SIDE_NAMES[SIDE_FOR_SIGN[winner]]} line from here')
            return

        waited = now - self.stage_since
        if waited > self.get_parameter('intersection.give_up_s').value:
            # Same reasoning as the light's give-up, and the same two causes
            # to tell apart: no messages at all means the detector never
            # armed or never ran, while messages that never formed a streak
            # means SIFT is matching intermittently and the thresholds want
            # looking at.
            recent = [SIGN_NAMES.get(v, v) for v in self.fresh_votes()]
            heading = ''
            if self.get_parameter('intersection.heading.enabled').value:
                want = self.get_parameter('intersection.heading.center_deg').value
                tol = self.get_parameter('intersection.heading.tolerance_deg').value
                yaw_txt = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                heading = (f', {self.sign_rejected} dropped on heading '
                           f'(facing {yaw_txt}, want {want:+.0f}+/-{tol:.0f})')
            why = (f'{self.sign_msgs} sign messages{heading}, '
                   f'last {len(recent)}: {recent}')
            self.get_logger().warn(
                f'no sign confirmed after {waited:.0f} s ({why})')
            # Carry on rather than stop. The same reasoning as the traffic
            # light's give-up: the clock does not pause for a detector that
            # failed, and stopping here loses every mission after this one as
            # well as this one. Steering stays on the mean of both lines,
            # which is what the robot was already doing.
            self.after_junction('gave up on the sign - carrying on')

    def accumulate_turn(self):
        """Add this tick's rotation to the running total."""
        if self.yaw is None:
            return
        if self.turn_last_yaw is not None:
            self.turn_accum += (self.yaw - self.turn_last_yaw + 180.0) % 360.0 - 180.0
        self.turn_last_yaw = self.yaw

    def turned_so_far(self):
        """Degrees turned since the turn began, unwrapped, or None."""
        if self.turn_start_yaw is None:
            return None
        return self.turn_accum

    def turn_target(self):
        """Degrees to turn for the arrow that was confirmed; + is left."""
        side = SIDE_FOR_SIGN.get(self.decision)
        if side == FOLLOW_YELLOW:
            return self.get_parameter('intersection.turn.left_deg').value
        if side == FOLLOW_WHITE:
            return self.get_parameter('intersection.turn.right_deg').value
        return 0.0

    def tick_turn(self, now):
        """Turn the fixed amount the confirmed arrow calls for, then hand over."""
        self.set_armed(MISSION_NONE)
        # Set during the turn rather than at the hand-over, so detect_lane is
        # already fitting the chosen line by the time the turn ends. Does
        # nothing when follow_chosen_line is off.
        self.set_follow_side(self.chosen_side())
        # True throughout: control_lane ignores lane following while
        # avoid_active is set, but its hold_tick would publish zeros over the
        # turn if drive_enable went false.
        self.set_driving(True)

        self.accumulate_turn()

        target = self.turn_target()
        turned = self.turned_so_far()
        if turned is None:
            self.get_logger().warn(
                'no odometry, so the turn cannot be measured - going on', once=True)
            self.finish_turn('no odometry')
            return

        remaining = target - turned
        # Signed, so this is "has it gone far enough in the intended
        # direction", not "is it close to the right amount" - overshooting
        # stops the turn rather than reversing it.
        if (target >= 0 and remaining <= 0) or (target < 0 and remaining >= 0):
            self.finish_turn(f'turned the {abs(target):.0f} deg asked for')
            return

        rate = self.get_parameter('intersection.turn.rate').value
        # Never more than half the turn, or a five degree turn would be spent
        # entirely crawling.
        slow = min(self.get_parameter('intersection.turn.slow_within_deg').value,
                   abs(target) * 0.5)
        if slow > 0 and abs(remaining) < slow:
            rate *= max(abs(remaining) / slow, 0.25)
        angular = rate if target >= 0 else -rate
        self.set_avoid(True, angular=angular,
                       linear=self.get_parameter('intersection.turn.speed').value)

        waited = now - self.stage_since
        if waited > self.get_parameter('intersection.turn.give_up_s').value:
            self.get_logger().warn(
                f'turned {turned:+.0f} of {target:+.0f} deg in {waited:.0f} s '
                f'- going on anyway')
            self.finish_turn('turn timed out')

    def finish_turn(self, why):
        """Stop turning and hand the wheel back to lane following."""
        turned = self.turned_so_far()
        # One last zero command, so control_lane is not left relaying the
        # rotation it was given on the tick before avoid_active went false.
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        self.after_junction(f'{why} after turning {turned:+.0f} deg'
                            if turned is not None else why)

    def chosen_side(self):
        """The line to steer by, or auto when the arrow only picked a turn."""
        if self.construction_seen:
            return FOLLOW_AUTO
        if not self.get_parameter('intersection.follow_chosen_line').value:
            return FOLLOW_AUTO
        return SIDE_FOR_SIGN.get(self.decision, FOLLOW_AUTO)

    def after_junction(self, why):
        """Where the wheel goes once the junction is behind us."""
        if self.get_parameter('construction.enabled').value:
            self.reset_votes()
            self.go(STAGE_DRIVE_TO_CONSTRUCTION, why)
        else:
            self.go(STAGE_FOLLOW_SIDE, why)

    def tick_follow_side(self):
        """Just drive. Nothing is being looked for."""
        self.set_armed(MISSION_NONE)
        self.set_driving(True)
        self.set_follow_side(self.chosen_side())
        self.set_avoid(False)

    def tick_drive_to_construction(self, now):
        """Carry on down the course until the construction sign is read."""
        self.set_driving(True)
        self.set_avoid(False)
        # Still steering by whatever the junction chose. Going back to both
        # lines happens when the sign is read, not on the way to it.
        self.set_follow_side(self.chosen_side())
        self.set_armed(MISSION_CONSTRUCTION)

        winner, votes = self.sign_majority({SIGN_CONSTRUCTION})
        if winner is not None:
            # Here is where the steering goes back to both lines.
            self.construction_seen = True
            self.set_follow_side(FOLLOW_AUTO)
            self.get_logger().info(
                f'construction sign confirmed: {votes} of the last '
                f'{len(self.fresh_votes())} reports '
                f'({self.sign_msgs} seen, {self.sign_rejected} ignored on heading)')
            if self.get_parameter('construction.stop_after_detect').value:
                self.stop('construction sign - stopping here, as asked')
            elif self.get_parameter('construction.avoid.enabled').value:
                self.go(STAGE_AVOID_CONSTRUCTION,
                        'construction sign - threading the boards')
            else:
                self.go(STAGE_FOLLOW_SIDE, 'construction sign read')
            return

        waited = now - self.stage_since
        if waited > self.get_parameter('construction.give_up_s').value:
            recent = [SIGN_NAMES.get(v, v) for v in self.fresh_votes()]
            yaw = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            self.get_logger().warn(
                f'no construction sign after {waited:.0f} s '
                f'({self.sign_msgs} reports, {self.sign_rejected} dropped on '
                f'heading, facing {yaw}, last {len(recent)}: {recent}) '
                f'- carrying on')
            self.go(STAGE_FOLLOW_SIDE, 'gave up on the construction sign')

    def tick_avoid_construction(self, now):
        """Thread the boards by hugging the line on the open side.

        No avoidance controller, no open-loop turns, no odometry. The boards
        alternate which half of a double-width corridor they block, and
        hugging a line is already how this stack drives half a corridor -
        detect_lane picks the line out and control_lane steers to it with
        gains that have been tuned on the course.

        Measured against the course texture: hugging white aims at x 1.752
        and the gap the first board leaves is centred on 1.746; hugging
        yellow aims at 1.517 against a gap centred on 1.506. Six and eleven
        millimetres out, from a mechanism built for a different mission.
        """
        self.set_driving(True)
        self.set_avoid(False)
        self.set_armed(MISSION_CONSTRUCTION)
        # White first: the first board blocks the left half, so the right
        # half is the one to be in.
        side = FOLLOW_WHITE
        self.set_follow_side(side)

        if (self.blocked is not None
                and self.blocked & BLOCKED_AHEAD
                and self.get_parameter(
                    'construction.avoid.stop_at_obstacle').value):
            self.stop(f'board across the {SIDE_NAMES[side]} line - '
                      f'stopping here, as asked')
            return

        waited = now - self.stage_since
        if waited > self.get_parameter('construction.avoid.give_up_s').value:
            # Two different faults look the same from here, so say which:
            # no report at all means detect_obstacle never armed or never
            # ran, while reports that stayed clear means the windows are
            # looking somewhere the boards are not.
            seen = 'no report yet' if self.blocked is None else f'last {self.blocked}'
            self.get_logger().warn(
                f'still in the construction zone after {waited:.0f} s '
                f'({seen}) - carrying on')
            self.go(STAGE_FOLLOW_SIDE, 'gave up threading the boards')

    def tick_stopped(self):
        self.set_armed(MISSION_NONE)
        self.set_driving(False)
        self.set_follow_side(FOLLOW_AUTO)
        self.set_avoid(False)


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

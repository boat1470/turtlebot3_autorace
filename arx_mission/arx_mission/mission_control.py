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

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Float32, Float64, UInt8
from tf2_ros import Buffer, TransformListener

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
# ARX: the example gives every sign its own Enum starting at 1, so its
# parking sign is also 1 and collides with the construction board. One node
# reading them all cannot have that, so parking gets the next free number.
SIGN_PARKING = 4
SIGN_TUNNEL = 5

SIGN_NAMES = {SIGN_CONSTRUCTION: 'construction',
              SIGN_LEFT: 'left', SIGN_RIGHT: 'right',
              SIGN_PARKING: 'parking', SIGN_TUNNEL: 'tunnel'}

# /arx/follow_side - must match detect_lane.py. Which lane line detect_lane
# should steer by once the junction has been decided: the yellow one on the
# left, the white one on the right, or the usual mean of the two.
FOLLOW_AUTO = 0
FOLLOW_YELLOW = 1
FOLLOW_WHITE = 2
# ARX: detect_lane's yellow pair - both walls yellow, as on the way into the
# parking lot. Must match detect_lane.py.
FOLLOW_YELLOW_PAIR = 3

SIDE_FOR_SIGN = {SIGN_LEFT: FOLLOW_YELLOW, SIGN_RIGHT: FOLLOW_WHITE}
SIDE_NAMES = {FOLLOW_AUTO: 'auto', FOLLOW_YELLOW: 'yellow', FOLLOW_WHITE: 'white'}

# /arx/armed_mission - tells a detector to do its work, and every other
# detector to idle. On the competition Raspberry Pi they cannot all run at
# once, and nothing downstream reads a detector that is not armed anyway.
MISSION_NONE = 0
MISSION_TRAFFIC_LIGHT = 1
MISSION_INTERSECTION = 2
MISSION_CONSTRUCTION = 3
MISSION_PARKING = 4
MISSION_LEVEL = 5
MISSION_TUNNEL = 6

STAGE_WAIT_GREEN = 'wait_green'
STAGE_DRIVE_TO_SIGN = 'drive_to_sign'
STAGE_TURN = 'turn'
STAGE_LEFT_SCRIPT = 'left_script'
STAGE_ARC = 'sign_arc'
STAGE_FOLLOW_SIDE = 'follow_side'
STAGE_DRIVE_TO_CONSTRUCTION = 'drive_to_construction'
STAGE_AVOID_CONSTRUCTION = 'avoid_construction'
STAGE_EDGE_PAST_BOARD = 'edge_past_board'
STAGE_DRIVE_TO_PARKING = 'drive_to_parking'
STAGE_DRIVE_TO_LOT = 'drive_to_lot'
STAGE_PICK_BAY = 'pick_bay'
STAGE_PARK_IN_BAY = 'park_in_bay'
STAGE_DRIVE_TO_LEVEL = 'drive_to_level'
STAGE_APPROACH_BAR = 'approach_bar'
STAGE_WAIT_BAR = 'wait_bar'
STAGE_DRIVE_TO_TUNNEL = 'drive_to_tunnel'
STAGE_TUNNEL_NAV = 'tunnel_nav'

# /detect/lane_state, from detect_lane: which line it is steering by.
LANE_STATE_FOR_SIDE = {FOLLOW_YELLOW: 1, FOLLOW_WHITE: 3}
# Both lines in view. detect_lane publishes 0 for neither.
LANE_STATE_BOTH = 2
STAGE_STOPPED = 'stopped'

# Every stage `start_stage` will accept (ARX). Written out rather than
# collected from the constants above so that adding a stage without deciding
# whether a run may start in it is a deliberate act.
# ARX: not a stage of its own - a start_stage that becomes drive_to_parking
# with the parking sign taken as read and both lines as held, so the run
# begins at "waiting for the yellow line to drop out" before 17B.
START_PARKING_17B = 'parking_17b'

STAGES = frozenset({
    START_PARKING_17B,
    STAGE_WAIT_GREEN,
    STAGE_DRIVE_TO_SIGN,
    STAGE_TURN,
    STAGE_FOLLOW_SIDE,
    STAGE_DRIVE_TO_CONSTRUCTION,
    STAGE_AVOID_CONSTRUCTION,
    STAGE_EDGE_PAST_BOARD,
    STAGE_DRIVE_TO_PARKING,
    STAGE_DRIVE_TO_LOT,
    STAGE_PICK_BAY,
    STAGE_PARK_IN_BAY,
    STAGE_DRIVE_TO_LEVEL,
    STAGE_APPROACH_BAR,
    STAGE_WAIT_BAR,
    STAGE_DRIVE_TO_TUNNEL,
    STAGE_TUNNEL_NAV,
    STAGE_STOPPED,
})

# /detect/obstacle, from detect_obstacle: which of the three strips ahead of
# the robot has a board in it. Relative to the robot, not to the corridor -
# a robot hugging the white line is already in the right half, so a board
# blocking that half is straight in front of it.
# /detect/obstacle_side, from detect_obstacle: which side of the robot has
# something standing next to it. The robot's own left and right, not the
# course's - a robot facing into the parking lot has the bay at the larger x
# on its LEFT, and naming these after the course is the mistake that once put
# the construction stage into the sign beside the track.
# /detect/level_bar, from detect_level: the level crossing bar. NONE is not
# the same as UP - it means the detector has nothing it is willing to call a
# bar, which at close range it reports alternately with UP as the row
# grouping loses a band. Only DOWN stops the robot.
BAR_NONE = 0
BAR_UP = 1
BAR_DOWN = 2
BAR_NAMES = {BAR_NONE: 'nothing', BAR_UP: 'up', BAR_DOWN: 'down'}

SIDE_CLEAR = 0
SIDE_LEFT = 1
SIDE_RIGHT = 2
SIDE_NAMES = {SIDE_LEFT: 'left', SIDE_RIGHT: 'right'}

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
        # ARX: look for the left arrow only. Right reports are not counted; a
        # left is taken on left_only.confirm_votes left reports in the window,
        # and if odom comes round to left_only.right_at_deg (turning left,
        # inside the heading gate) without one, the arrow is taken as right.
        # The right arm far off is a 7-10 px disc that was read too seldom,
        # or the lowered left arm read in its place (1023_2258, 1023_2324).
        self.declare_parameter('intersection.left_only.enabled', False)
        self.declare_parameter('intersection.left_only.confirm_votes', 3)
        self.declare_parameter('intersection.left_only.right_at_deg', 155.0)
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

        # A left arrow is driven by a fixed script instead of the turn below
        # (ARX): on the real course the yellow line of the left branch is not
        # in frame where the arrow is confirmed, so there is nothing to hand
        # over to. Face an absolute heading, forward, turn left, forward,
        # and only then steer by the yellow line. A right arrow is unchanged.
        #
        # The heading is a world angle - see spawn_yaw_deg - so it is right
        # only if the robot was brought up facing along the start line.
        self.declare_parameter('intersection.left_script.enabled', False)
        # ARX: the quicker left turn. On a left arrow, hold one fixed arc -
        # linear speed and turn rate - for up to timeout_s, or until
        # detect_lane has both lines of the branch (lane_state 2 for
        # both_ticks ticks in a row), whichever is first, and then follow
        # the lane by both lines. Takes precedence over left_script, which
        # faces a heading, drives, turns and drives again, and took 18 s.
        self.declare_parameter('intersection.left_arc.enabled', False)
        self.declare_parameter('intersection.left_arc.speed', 0.10)      # m/s
        self.declare_parameter('intersection.left_arc.rate', 0.6)        # rad/s, + is left
        self.declare_parameter('intersection.left_arc.timeout_s', 2.0)
        self.declare_parameter('intersection.left_arc.both_ticks', 3)
        self.declare_parameter('intersection.left_arc.min_s', 0.5)
        # ARX: the same for a right arrow, mirrored. rate is a magnitude; the
        # arc turns right (negative angular).
        self.declare_parameter('intersection.right_arc.enabled', False)
        self.declare_parameter('intersection.right_arc.speed', 0.14)
        self.declare_parameter('intersection.right_arc.rate', 0.7)
        self.declare_parameter('intersection.right_arc.timeout_s', 2.0)
        self.declare_parameter('intersection.right_arc.both_ticks', 3)
        self.declare_parameter('intersection.right_arc.min_s', 0.5)
        # ARX: a first beat before the right arc. The right arrow is
        # confirmed while the robot is still in the left-hand curve before
        # the junction (2A), so turning right there straight away swings it
        # back mid-curve. First carry on round to align_heading_deg (odom),
        # on an arc of its own, then turn right. align: false skips it.
        self.declare_parameter('intersection.right_arc.align', False)
        self.declare_parameter('intersection.right_arc.align_heading_deg', 0.0)
        self.declare_parameter('intersection.right_arc.align_tolerance_deg', 5.0)
        self.declare_parameter('intersection.right_arc.align_speed', 0.14)
        self.declare_parameter('intersection.right_arc.align_rate', 0.7)
        self.declare_parameter('intersection.right_arc.align_timeout_s', 2.0)
        # ARX: then straight in on that heading for straight_m (odom), before
        # the turn - into the junction, so the right turn starts where the
        # right branch does. 0 skips it.
        self.declare_parameter('intersection.right_arc.straight_m', 0.10)
        self.declare_parameter('intersection.right_arc.straight_speed', 0.14)
        self.declare_parameter('intersection.right_arc.straight_timeout_s', 3.0)
        # ARX: end the right arc at this odom heading rather than on
        # timeout_s, which then only backs it up. The angle comes out the
        # same however late the sign was confirmed. use_end_heading: false
        # keeps the time.
        self.declare_parameter('intersection.right_arc.use_end_heading', False)
        # ARX: after the right arc, steer by the white line rather than both,
        # until odom crosses hug_until_deg turning the hug_release_turn way
        # ('right' or 'left'), then go back to both lines. The right branch
        # runs 5B, the U-turn 6A left round the island and the U-turn 7A
        # right - and heading passes 180 twice, once in each U-turn, the
        # first turning left and the second turning right. hug_timeout_s
        # lets go regardless. hug_white: false keeps both lines throughout.
        self.declare_parameter('intersection.right_arc.hug_white', False)
        self.declare_parameter('intersection.right_arc.hug_until_deg', 180.0)
        self.declare_parameter('intersection.right_arc.hug_release_turn', 'right')
        self.declare_parameter('intersection.right_arc.hug_timeout_s', 25.0)
        self.declare_parameter('intersection.right_arc.end_heading_deg', -70.0)
        self.declare_parameter('intersection.left_script.heading_deg', 180.0)
        self.declare_parameter('intersection.left_script.heading_tolerance_deg', 3.0)
        self.declare_parameter('intersection.left_script.first_m', 0.30)
        # Relative, + is left.
        self.declare_parameter('intersection.left_script.turn_deg', 90.0)
        self.declare_parameter('intersection.left_script.second_m', 0.30)
        self.declare_parameter('intersection.left_script.rate', 0.4)       # rad/s
        self.declare_parameter('intersection.left_script.speed', 0.05)     # m/s
        self.declare_parameter('intersection.left_script.slow_within_deg', 10.0)
        # Per step. Catches odometry that never arrives or a wheel that is
        # held; the script then moves on to the next step rather than stall.
        self.declare_parameter('intersection.left_script.step_give_up_s', 15.0)
        # Stop where the script ends instead of handing over to the yellow
        # line. A bench aid, like stop_after_detect: it fails the run.
        self.declare_parameter('intersection.left_script.stop_after', False)
        # What to steer by once the script is done: true pins detect_lane to
        # the yellow line until the construction sign, whatever
        # follow_chosen_line says; false hands back the mean of both lines.
        self.declare_parameter('intersection.left_script.follow_yellow_after', True)
        # The same choice for a right arrow, after its turn: true pins the
        # white line (when follow_chosen_line is on), false is the mean of
        # both lines.
        self.declare_parameter('intersection.right_follow_white_after', True)

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
        # ARX: a bench aid. Stop in drive_to_construction the moment the
        # white line drops out of the lane estimate - lane_state 0 or 1 for
        # white_lost_ticks ticks in a row - so where and why it went can be
        # looked at with the robot standing there. Ends the run.
        self.declare_parameter('construction.stop_when_white_lost', False)
        self.declare_parameter('construction.white_lost_ticks', 3)
        # ARX: instead of stopping, drive a fixed arc of radius_m (angular =
        # speed / radius_m, the side's way) until both lines are back - lane_state
        # 2 for back_ticks ticks, after min_s - or timeout_s is up (0 = never), then
        # follow the lane again. Takes precedence over stop_when_white_lost.
        self.declare_parameter('construction.white_lost_arc.enabled', False)
        self.declare_parameter('construction.white_lost_arc.side', 'right')
        self.declare_parameter('construction.white_lost_arc.radius_m', 0.25)
        self.declare_parameter('construction.white_lost_arc.speed', 0.10)
        self.declare_parameter('construction.white_lost_arc.back_ticks', 2)
        # What ends it: 'both' - lane_state 2; 'yellow' - 1 or 2, the yellow
        # line in view with or without white; 'yellow_pair' - two yellow lines
        # a lane apart (detect_lane in FOLLOW_YELLOW_PAIR, lane_state 2).
        self.declare_parameter('construction.white_lost_arc.until', 'both')
        self.declare_parameter('construction.white_lost_arc.min_s', 0.3)
        self.declare_parameter('construction.white_lost_arc.timeout_s', 3.0)
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

        # Swinging round a board that blocks the line being hugged.
        #
        # An arc out to `swing_deg` off the corridor and a mirrored arc back.
        # Two arcs of radius r through the same angle move the robot 2r
        # sideways and 2r forward, so one number sets both, and r = 0.10
        # gives the 0.19 m of sideways the measured gap beside barrier_2
        # needs, inside the 0.31 m of corridor before it.
        #
        # Closed on yaw rather than on the LiDAR: a board 0.25 m across
        # leaves only 0.11 m of surface to follow once the robot's own width
        # is taken off, which is not enough for a distance controller to
        # settle in. Yaw against the heading the robot had while still lane
        # following says the same thing the white line going from upright to
        # flat in the camera would say, and detect_lane cannot say it -
        # it fits x as a function of y from the bottom of the frame up,
        # which a flat line is not.
        self.declare_parameter('construction.avoid.edge.enabled', True)
        self.declare_parameter('construction.avoid.edge.radius_m', 0.12)
        # Swing round the next board straight after the first, without going
        # back to hugging a line in between. Measured on the run that
        # threaded all three boards: the closest approach to barrier_3 was
        # zero - a touch - and it happened at yaw 15 degrees, which is to say
        # the robot was still crossing when it arrived. The delay was going
        # back to the yellow line, letting control_lane settle on it, and
        # then waiting for the LiDAR to see the next board at 0.25 m, all
        # inside the 0.36 m gap between the two boards.
        self.declare_parameter('construction.avoid.edge.double_swerve', True)
        self.declare_parameter('construction.avoid.edge.second_radius_m', 0.10)
        # Straight run between the two swings. Not optional: coming straight
        # back carries the robot into the board's own x range while it is
        # still alongside its 0.10 m depth, which the simulation hits at any
        # gap under 0.30 m for the old radius.
        self.declare_parameter('construction.avoid.edge.between_m', 0.20)
        # A straight run at the far end of the second swing, while the robot
        # is across the corridor and pointing at the side it came from. Every
        # centimetre here is a centimetre back towards that side, bought
        # without turning any further - which is what the boards leave no
        # room for.
        self.declare_parameter('construction.avoid.edge.cross_m', 0.10)
        # Straight run right after the on-the-spot turn, while the robot
        # is pointing across the corridor. This is where the sideways
        # move comes from when the turn itself made none.
        self.declare_parameter('construction.avoid.edge.out_m', 0.10)
        # Turn on the spot instead of arcing round.
        #
        # An arc goes forward while it turns, which spends the gap in front
        # of the board getting closer to it, and swings the outside of the
        # robot wide - measured at 72 to 323 mm past the yellow line
        # depending on the radius, in a corridor only 0.485 m across.
        # Turning on the spot spends neither: the sideways move becomes the
        # straight run between the two turns and nothing else, so it is one
        # number and it is exact.
        #
        # linear.x is zero while turning, which is not the same as the robot
        # having stopped - 90 degrees at this rate takes under two seconds,
        # against the 30 the rules allow before a run is ended.
        self.declare_parameter('construction.avoid.edge.in_place', True)
        self.declare_parameter('construction.avoid.edge.in_place_rate', 0.6)
        # Only for the first board in the zone. That is the one the robot
        # arrives at square and at speed, with the least room to spare -
        # measured at 13.4 mm with an arc against 35.3 turning on the spot.
        # After it the robot is already threading and an arc is quicker,
        # which matters against a 300 second limit.
        self.declare_parameter('construction.avoid.edge.in_place_first_only', True)
        self.declare_parameter('construction.avoid.edge.speed', 0.04)
        self.declare_parameter('construction.avoid.edge.swing_deg', 90.0)
        # ARX: swing to fixed odom headings instead of swing_deg relative to
        # wherever the robot pointed when it began. Relative carries any skew
        # lane following left at that moment into every heading after it;
        # absolute does not. Off by default, so the simulator's tuning is
        # unchanged. corridor_heading_deg is the way the zone runs, and
        # swing_heading_deg the way to point while getting round the first
        # board - which also decides the side, overriding the hugged line.
        # The second swing of a double swerve mirrors it about the corridor.
        self.declare_parameter('construction.avoid.edge.absolute', False)
        self.declare_parameter('construction.avoid.edge.corridor_heading_deg', 90.0)
        self.declare_parameter('construction.avoid.edge.swing_heading_deg', 180.0)
        # ARX: absolute only. Ease the on-the-spot turn over its last
        # stretch, down to a quarter of in_place_rate, and steer back onto
        # swing_heading_deg while driving out. At full rate into the target
        # the real robot carried on 11 degrees past it after the stop - the
        # odom it stops on and the stop it is sent both cross Wi-Fi - and
        # then drove `out` diagonally, towards the board ahead.
        self.declare_parameter('construction.avoid.edge.slow_within_deg', 20.0)
        # There is no tolerance parameter for the end of a swing back. It
        # finishes when the heading crosses the one it started from, which
        # is a sign change and cannot be stepped over.
        #
        # It used to finish when the heading came within settle_deg, and
        # that is a window 2 x settle_deg wide that the robot has to land a
        # tick inside. It turns 1.9 degrees a tick at radius 0.12 and 3.4
        # turning on the spot, so the window is only a few steps across;
        # miss it and the command does not change, the robot keeps turning
        # the same way, and it can never come back. A run ended that way
        # with turned stuck at -20 degrees against a 12 degree window - past
        # the heading it was aiming for, still turning away from it.
        self.declare_parameter('construction.avoid.edge.give_up_s', 25.0)
        # ARX: three beats instead of phases 1-6. After the spin to
        # swing_heading_deg, one arc back the other way to arc1_end_deg - over
        # the board and down past it - then an arc the first way back to
        # corridor_heading_deg, which also takes the robot past the next board.
        # Both ends on odom headings, mirrored when the swing goes right.
        # speed is cut on each arc so the outer wheel stays inside wheel_max:
        # v <= wheel_max / (1 + half_track / r).
        self.declare_parameter('construction.avoid.edge.three_beat.enabled', False)
        self.declare_parameter('construction.avoid.edge.three_beat.speed', 0.15)
        self.declare_parameter('construction.avoid.edge.three_beat.arc1_radius_m', 0.28)
        self.declare_parameter('construction.avoid.edge.three_beat.arc1_end_deg', 30.0)
        self.declare_parameter('construction.avoid.edge.three_beat.arc2_radius_m', 0.20)
        self.declare_parameter('construction.avoid.edge.three_beat.wheel_max', 0.21)
        self.declare_parameter('construction.avoid.edge.three_beat.half_track', 0.08)
        # End each arc this many ticks of turning before its heading, not on
        # the first tick past it: a tick of 0.54 rad/s at 10 Hz is 3 degrees,
        # and one past the first arc's end lands the robot 3 cm lower.
        self.declare_parameter('construction.avoid.edge.three_beat.end_lead_ticks', 0.5)

        # --- mission 4, the parking sign ---------------------------------
        # Same shape as the other two sign hunts. What is different is the
        # second condition: the board only counts when detect_lane has both
        # lines, which is what says the robot is back on an ordinary
        # single-width lane rather than still in the construction corridor,
        # where the boards and the double width make a stray match likely.
        self.declare_parameter('parking.enabled', True)
        self.declare_parameter('parking.window', 7)
        self.declare_parameter('parking.confirm_votes', 5)
        # Six, like the construction board and for the same reason: this one
        # is also passed side-on while the robot drives, not stared at.
        self.declare_parameter('parking.vote_max_age_s', 6.0)
        # Stop where the sign was found. The debug step; nothing is built on
        # top of it yet.
        self.declare_parameter('parking.stop_after_detect', True)
        self.declare_parameter('parking.give_up_s', 120.0)
        # The board sits at (0.74, 1.95) turned -90 degrees in the world
        # file, the same as the others, so it can only be read along the
        # world x axis. 180 is looking towards -x.
        self.declare_parameter('parking.heading.enabled', True)
        self.declare_parameter('parking.heading.center_deg', 180.0)
        self.declare_parameter('parking.heading.tolerance_deg', 45.0)
        # Both lines in view. Judged from the two reliability topics, not
        # from lane_state: while follow_side pins the steering to white,
        # lane_state is 3 by definition and can never say both.
        self.declare_parameter('parking.require_both_lines', True)
        # Reliability counts how much of the frame a line fills and moves by
        # 5 a frame, so 50 is about ten frames of having seen it.
        self.declare_parameter('parking.min_reliability', 50)
        # ARX: ...and for this long without a break, before leaving the white
        # line for the yellow one. The sign is remembered once confirmed, so
        # it does not have to be in view at the same moment - its votes
        # expire after vote_max_age_s, and the lines may come right later.
        self.declare_parameter('parking.both_lines_s', 2.0)
        # ARX: a bench aid. Once the parking sign is read and both lines have
        # held both_lines_s, do not turn in: carry on, and stop the moment the
        # yellow line then drops out - yellow reliability under min_reliability
        # for yellow_lost_ticks ticks in a row - so where it went can be looked
        # at with the robot standing there. Reliability, not lane_state: the
        # steering is pinned to white here, so lane_state is 3 regardless.
        self.declare_parameter('parking.stop_when_yellow_lost', False)
        self.declare_parameter('parking.yellow_lost_ticks', 3)
        # ARX: instead of stopping there, turn in at 17B: pre_m straight on,
        # then one left arc of radius_m (angular = speed / radius_m) until odom
        # reads end_heading_deg, then drive_to_lot. R 0.25 from the lane
        # centre (x 3.75) is the arc round 17B's own centre, and comes out
        # mid-way between the lot's two yellow walls. Wins over the stop.
        self.declare_parameter('parking.left_arc.enabled', False)
        self.declare_parameter('parking.left_arc.pre_m', 0.0)
        self.declare_parameter('parking.left_arc.radius_m', 0.25)
        self.declare_parameter('parking.left_arc.speed', 0.12)
        self.declare_parameter('parking.left_arc.end_heading_deg', -90.0)
        self.declare_parameter('parking.left_arc.timeout_s', 6.0)
        # After the arc: follow the yellow pair down the way in, and lock
        # end_heading_deg once the pair is lost for pair_lost_ticks - or if it
        # has not come into view within pair_wait_s of the arc ending.
        self.declare_parameter('parking.left_arc.pair_lost_ticks', 3)
        self.declare_parameter('parking.left_arc.pair_wait_s', 1.0)

        # The parking lot itself (ARX). Measured off the course texture, which
        # is an exact map of the track - see notes/track_geometry.py:
        #
        #   main road        white y 1.879, yellow y 1.625
        #   yellow line      stops at x 0.631 and starts again at x 0.381
        #   pocket mouth     x 0.39 to 0.62 at y 1.008, so 0.23 m wide
        #   pocket           x 0.135 to 0.877, y 0.500 to 1.008
        #   two bays         x 0.631-0.877 and x 0.135-0.381, 0.246 m each,
        #                    split off a middle column that is the way in
        #
        # Nothing marks the turn-in. traffic_pl_left, the only other board
        # there, stands at (0.50, 1.90) turned to face -y, which is the robot
        # coming back OUT of the pocket - it is the exit sign, not the
        # entrance. What does mark the entrance is the gap in the yellow
        # line, which is the same thing the mouth is.
        self.declare_parameter('lot.enabled', True)
        # How many ticks of lane_state 0 count as the yellow line having run
        # out. detect_lane publishes at about the rate this ticks, so three
        # is roughly a third of a second - 13 mm at the speed below.
        self.declare_parameter('lot.confirm_ticks', 3)
        # How far to carry on after that, straight.
        self.declare_parameter('lot.lead_m', 0.30)
        # Driven by this node rather than by lane following, so it needs its
        # own speed. The same creep the construction swings use.
        self.declare_parameter('lot.speed', 0.04)
        # ARX: while holding the heading, v is cut so the outer wheel
        # (v + |omega| x half_track) stays inside wheel_max - a 0.22 is
        # refused by the OpenCR outright.
        self.declare_parameter('lot.wheel_max', 0.21)
        self.declare_parameter('lot.half_track', 0.08)
        # ARX: lane-following speed ceiling while turning off the road into
        # the lot, published on /control/max_vel until the corridor heading is
        # held. The turn follows a yellow arc of 0.11 m radius, and at the
        # road's speed the robot ran wide of it - control_lane turns at
        # kp x error whatever the speed, so a slower robot turns tighter for
        # the same error. 0 leaves the speed alone. control.max_vel is what is
        # put back afterwards, read from the same mission.yaml.
        self.declare_parameter('lot.approach_max_vel', 0.0)
        # ...but only once the turn has begun: this many degrees of it. From
        # the start of the stage, the robot crawled the whole road to 17B.
        self.declare_parameter('lot.slow_from_deg', 10.0)
        # Steer by both yellow walls on the way in (detect_lane's yellow
        # pair) rather than the left one plus half a lane. Only until the
        # corridor heading is held, as before; the way out stays on yellow.
        self.declare_parameter('lot.yellow_pair', False)
        # ...and once both yellow walls have been seen for this many ticks in
        # a row (lane_state 2 in the yellow pair mode, which reaches 2 no
        # other way), the slow-down ends: the robot is in the way in, lined up
        # between them, and the turn it was slowed for is done.
        self.declare_parameter('lot.pair_release_ticks', 3)
        self.declare_parameter('control.max_vel', 0.15)
        # ARX: the speed ceiling while a sign is being hunted - drive_to_sign,
        # drive_to_construction, drive_to_parking. detect_sign needs 5 of the
        # last 7 reports to agree, and at speed a board can pass before they
        # do; slowing these stages alone is cheaper than slowing the course.
        # 0 leaves the speed alone.
        self.declare_parameter('speed.sign_hunt', 0.0)
        self.max_vel_want = None       # this tick's override, set by a stage
        self.max_vel_sent = None
        self.max_vel_tick = 0
        # Take the wheel as soon as the robot is running straight down the
        # corridor, rather than waiting for the yellow line to end.
        #
        # Waiting cost 16 degrees, measured: the robot held x 0.486 on -89.8
        # for seconds, and then over the last 0.13 m - while the line was
        # ending but before lane_state would admit it - lane following swung
        # it to -77.4 and carried it out to x 0.553. Correcting afterwards
        # got the heading back, but only after the swing had happened.
        #
        # There is nothing lost by taking it early. The corridor is straight,
        # so lane following has no work to do in it; every steering command
        # it gives there is either noise or, at the end, wrong.
        #
        # Two conditions, because either alone is wrong. The turn is what
        # says the robot is in the corridor at all - the road before it is
        # just as straight, and holding a heading there would drive past the
        # entrance. The steadiness is what says the turn is finished.
        self.declare_parameter('lot.turn_in_deg', 60.0)
        # Steady means the heading has stayed inside this band for this many
        # ticks. Measured: the swing through the turn is tens of degrees, the
        # corridor run holds to under half a degree a second, and one second
        # of odometry is quiet enough to tell them apart without a rate
        # estimate - which at 10 Hz is mostly noise.
        self.declare_parameter('lot.steady_ticks', 10)
        self.declare_parameter('lot.steady_band_deg', 2.0)
        # rad/s of correction per degree of error, and its ceiling. Holding
        # rather than recovering now, so the errors it sees are small.
        self.declare_parameter('lot.kp', 0.02)
        self.declare_parameter('lot.max_angular', 0.5)
        self.declare_parameter('lot.stop_at_end', False)
        self.declare_parameter('lot.give_up_s', 60.0)

        # Which bay to park in (ARX). Read off /detect/obstacle_side while
        # standing still between the two of them, which is the one piece of
        # the 2018 example's detect_parking.py worth keeping: it scans 30
        # degrees either side of straight left and straight right against
        # half a metre. The rest of that file is dead reckoning - straight
        # 0.45, left 90, straight 1.0 - and this package reaches the same
        # place by following the yellow line instead.
        self.declare_parameter('bay.enabled', True)
        # Stand still this long before reading. The robot arrives with its
        # own drift still settling, and the scan is free, so there is no
        # reason to act on the first one.
        self.declare_parameter('bay.settle_s', 1.0)
        # Which to take when both are empty. Right, like the example, which
        # tested the right first - and in the simulator both ARE empty unless
        # full.launch.py blocked:= puts a robot in one.
        self.declare_parameter('bay.prefer', 'right')
        self.declare_parameter('bay.stop_after_pick', True)
        self.declare_parameter('bay.give_up_s', 20.0)

        # Backing into the bay and coming out again (ARX). Five phases:
        #
        #   0 turn 90 degrees on the spot, so the empty bay is behind
        #   1 reverse into it
        #   2 stand there
        #   3 drive out on a quarter circle, turning the same way again
        #   4 hold the heading until there is a line to follow, then hand back
        #
        # Reversing in rather than driving in, because the robot has to leave
        # again: parked nose-first it would have to reverse out blind down a
        # 0.25 m column, and parked tail-first the whole exit is one forward
        # arc.
        self.declare_parameter('bay.park.enabled', True)
        self.declare_parameter('bay.park.turn_deg', 90.0)
        self.declare_parameter('bay.park.turn_rate', 0.6)
        # ARX: phase 0 turns to an odom heading rather than through turn_deg,
        # so where the robot stopped facing does not carry into the bay. The
        # heading is the one that points the tail at the bay: facing up (180)
        # backs into the robot's left bay, facing down (0) into its right,
        # with the robot stopped facing -90 down the lot. Eased in over the
        # last slow_within_deg, down to a quarter of turn_rate.
        self.declare_parameter('bay.park.absolute', False)
        self.declare_parameter('bay.park.face_left_deg', 180.0)
        self.declare_parameter('bay.park.face_right_deg', 0.0)
        self.declare_parameter('bay.park.slow_within_deg', 20.0)
        # ARX: the reverse into the bay at its own speed (0 = bay.park.speed),
        # ended reverse_lead_ticks of travel early so the stop lands on
        # into_*_m: odometry reports a tick behind the command.
        self.declare_parameter('bay.park.reverse_speed', 0.0)
        self.declare_parameter('bay.park.reverse_lead_ticks', 1.5)
        # ARX: the way out. exit_speed for the arc and the straight after it
        # (0 = bay.park.speed), cut on the arc so the outer wheel stays inside
        # lot.wheel_max. With exit_absolute the arc ends on exit_heading_<bay>_deg
        # rather than after turn_deg, half a tick early, and the straight
        # holds that heading until the yellow line is in view - yellow
        # reliability over parking.min_reliability for rejoin_ticks - rather
        # than any lane_state, which the lot's white dashes would satisfy.
        self.declare_parameter('bay.park.exit_speed', 0.0)
        self.declare_parameter('bay.park.exit_absolute', False)
        # One per bay: the left bay's arc comes round from 180 turning right,
        # the right bay's from 0 turning left, so they finish from opposite
        # sides of the way-out heading and land differently.
        self.declare_parameter('bay.park.exit_heading_left_deg', 90.0)
        self.declare_parameter('bay.park.exit_heading_right_deg', 90.0)
        self.declare_parameter('bay.park.speed', 0.04)
        # Two distances, not one, because the robot does not stop in the
        # middle of the column. It stops at x 0.476 against a middle of
        # 0.506 - four runs agreeing to 3.5 mm - so the bay centres at 0.754
        # and 0.258 are 0.278 and 0.218 away. A single figure would leave 4 mm
        # at one end of a bay that only has 68 mm to give.
        self.declare_parameter('bay.park.into_left_m', 0.278)
        self.declare_parameter('bay.park.into_right_m', 0.218)
        # Stand still in the bay. Zero is what was asked for; the knob is
        # here because being seen stopped in the bay may be what the rules
        # actually score.
        self.declare_parameter('bay.park.hold_s', 0.0)
        # The quarter circle out. From the left bay's centre at (0.754,
        # 0.754) facing -x, turning right through 90 degrees on this radius
        # ends at (0.506, 1.002) facing +y - the middle of the mouth, pointed
        # up the corridor - so no straight run is needed before or after it.
        # omega is speed/radius, 0.161 rad/s here against a wheel ceiling of
        # 2.25 at this speed.
        self.declare_parameter('bay.park.radius_m', 0.248)
        # Creep on the exit heading until detect_lane has something again,
        # then hand back to it. Without this the wheel goes back at the top
        # of the arc, where the robot is still inside the pocket and there is
        # no line for another 50 mm.
        self.declare_parameter('bay.park.rejoin', True)
        self.declare_parameter('bay.park.rejoin_ticks', 3)
        # Once there is a line again, follow the YELLOW one until the robot is
        # pointing back down the road, and only then hand to the mean of both.
        #
        # Handing straight to the mean did not work. The robot came up the
        # corridor correctly to (0.469, 1.624), then turned past 180 to -139.7
        # and drove back down into the pocket. The corridor has a yellow wall
        # on BOTH sides and no white one, so detect_lane reports lane_state 1
        # whichever it locks onto, and at the top the yellow it was holding
        # curves away west - the robot followed it round and back in.
        #
        # Naming the line and the heading to leave on takes both guesses out:
        # the yellow line is the one that leads out of the pocket and onto the
        # road, and 180 is the road.
        self.declare_parameter('bay.park.leave_heading_deg', 180.0)
        self.declare_parameter('bay.park.leave_tolerance_deg', 20.0)
        self.declare_parameter('bay.park.give_up_s', 90.0)

        # The level crossing (ARX). No sign to hunt here - the bar itself is
        # what says the mission has begun - so this is one watch, one creep
        # and one wait.
        #
        # Measured walking in to the bar at (-0.85, 1.26) in 5 cm steps:
        #
        #   0.85 .. 0.65 m   bar up, the resting state
        #   0.60 m           the bar comes down
        #   0.55 .. 0.35 m   down, and it holds 13.1 s
        #   0.30 m           up again
        #
        # 13.1 s is 0.91 m at the speed this drives, against 0.30 m to cover,
        # so the robot reaches the stop with about nine seconds to spare.
        self.declare_parameter('level.enabled', True)
        self.declare_parameter('level.confirm_ticks', 3)
        # Only count "down" while facing the crossing's road - see heading_ok.
        # Range alone cannot tell the bar from a wall met in a curve.
        self.declare_parameter('level.heading.enabled', True)
        self.declare_parameter('level.heading.center_deg', 0.0)
        self.declare_parameter('level.heading.tolerance_deg', 20.0)
        # Not UP, but "anything other than DOWN". detect_level alternates
        # between up and nothing at close range - measured - and both mean the
        # road is clear. Waiting for a run of UP alone can stall on that
        # flicker.
        self.declare_parameter('level.clear_ticks', 5)
        # Where to stand. Nearer than 0.20 m the camera has nothing to look
        # at: it sits 0.076 m ahead of the model origin, which is 0.038 m
        # ahead of the robot's own nose, so at 0.06 m it is already past the
        # bar. 0.30 m leaves the bar filling a good part of the frame with
        # room to spare.
        self.declare_parameter('level.stop_m', 0.30)
        self.declare_parameter('level.speed', 0.04)
        # Up to the bar on the lane, capped at approach_max_vel, rather than
        # on a held heading at level.speed (false).
        self.declare_parameter('level.approach_follow_lane', True)
        self.declare_parameter('level.approach_max_vel', 0.10)
        # A cap on the creep, because /detect/level_range is only trustworthy
        # inside about 0.50 m: further out the detector often resolves three
        # of the four bands and the span it measures is between the wrong
        # two. The range is what aims; this is what stops a bad one running
        # the robot into the bar.
        self.declare_parameter('level.max_creep_m', 0.60)
        # What decides at the crossing. 'ultrasonic': /detect/level_range
        # alone, against a threshold per stage - slow under slow_m, stop at
        # stop_m, go once it is past clear_m. 'camera': /detect/level_bar
        # (down/up) as detect_level reports it. full.launch sets it from
        # level_source.
        self.declare_parameter('level.source', 'ultrasonic')
        self.declare_parameter('level.slow_m', 0.45)
        # ARX: with level.source camera, its own two distances: start creeping
        # once DOWN is seen at or inside camera.slow_m, stop at camera.stop_m.
        # The camera's range is an estimate off the bands' span and it goes
        # blind under ~0.20 m, so it does not share the ultrasonic's numbers.
        self.declare_parameter('level.camera.slow_m', 0.40)
        self.declare_parameter('level.camera.stop_m', 0.20)
        self.declare_parameter('level.clear_m', 0.20)
        # A range older than this is no range - the Pi node stopped, or the
        # network dropped it.
        self.declare_parameter('level.range_max_age_s', 0.5)
        self.declare_parameter('level.stop_at_bar', False)
        self.declare_parameter('level.give_up_s', 90.0)
        # Under thirty seconds, deliberately. The rules end a run when the
        # robot has not moved for that long, so a bar that never lifts has to
        # be given up on before then rather than after.
        self.declare_parameter('level.wait_give_up_s', 25.0)

        # The tunnel (ARX). Getting in is the whole of this stage; what to do
        # once inside is not built yet.
        #
        # There is no sign hunt and no upward range finder. The walls are
        # already in the world - tunnel_wall, four of them, 0.05 thick and
        # 0.25 tall, enclosing about 1.93 x 1.85 m with a 0.34 m way in - and
        # the LiDAR reads them straight off /detect/obstacle_side:
        #
        #   outside, y +0.60    left 1.081   right nothing
        #   outside, y +0.20    left 0.673   right nothing
        #   the mouth, y -0.10  left 0.169   right 0.120
        #   inside,  y -0.35    left 0.355   right 0.120
        #
        # Both sides at once is the signature worth waiting for. One side
        # alone is any prop standing near the road; both, at these distances,
        # is a corridor. It holds for about 0.5 m from the mouth, which at 10
        # Hz is a long time to confirm in.
        #
        # An upward range finder would say it more directly, and was asked
        # for. It would also mean forking the example's robot model, its
        # spawn launch and the simulator launch that includes it, because a
        # sensor has to live in the model and gz's DetachableJoint has to be
        # declared in the parent model as well. The LiDAR answers the same
        # question with nothing forked, and without depending on the light.
        self.declare_parameter('tunnel.enabled', True)
        # Stop steering by the lane when something comes inside this, in the
        # wide wedge ahead. The lane ends at the mouth - under the roof there
        # is no paint at all - so the last thing to do while there is still a
        # line is to take the heading it was holding and carry it in.
        # Which way the robot has to be pointing for any of this to count.
        #
        # Without it the test is "something inside 0.20 m ahead, then
        # something on the right", and that is true in a great many places. A
        # run came through mission 5 and called the tunnel while facing +0.7
        # degrees - travelling +x, on the level crossing road, ninety degrees
        # off the only heading the mouth can be entered from. The stop board
        # at (-1.35, 1.04) sits about 0.21 m off that line on the right, which
        # fits what the scan reported.
        #
        # It is the same mistake the construction stage made once, and every
        # sign hunt in this file has carried a heading gate since.
        #
        # ARX 2026-10-17: the front of the tunnel is now the tunnel board,
        # seen big enough (sign.tunnel.min_width_px) while facing -90 +/- 30 -
        # measured on the real course at -88.5. The gate is applied to the
        # sign reports in on_sign, like every other sign hunt.
        self.declare_parameter('tunnel.heading.enabled', True)
        self.declare_parameter('tunnel.heading.center_deg', -90.0)
        self.declare_parameter('tunnel.heading.tolerance_deg', 30.0)
        self.declare_parameter('tunnel.confirm_votes', 3)
        # Stop at the front of the tunnel instead of going in. For now, until
        # what to do inside is decided. The front is two things in order: the
        # board confirmed, and after it something within stop_ahead_m ahead
        # on the LiDAR. Something that close before the board does not count.
        self.declare_parameter('tunnel.stop_at_sign', True)
        self.declare_parameter('tunnel.stop_ahead_m', 0.25)
        # Readings in a row - /detect/front_range jumps for a scan now and
        # then (0.437 -> 0.242 -> 0.534 in one run), and one is not a wall.
        self.declare_parameter('tunnel.stop_ahead_ticks', 2)
        # Speed ceiling between the board and the stop. At 0.2 m/s the robot
        # covers 2 cm a tick and more between scans, so it would stop well
        # inside 0.15 m.
        self.declare_parameter('tunnel.after_sign_max_vel', 0.08)
        self.declare_parameter('tunnel.vote_max_age_s', 3.0)
        self.declare_parameter('tunnel.speed', 0.04)
        # Both walls at once, not the right one alone. The right is the
        # steadier of the two - a flat 0.120 m over sixty beams from the mouth
        # to the far end, while the left falls away to 1.106 m within half a
        # metre - but steadier is not the same as particular, and on its own
        # it accepted a sign by the road. Measured at the mouth: left 0.176,
        # right 0.140, together.
        self.declare_parameter('tunnel.need_both', True)
        self.declare_parameter('tunnel.confirm_ticks', 5)
        # Stop on getting in. The debug step, like every mission before it.
        self.declare_parameter('tunnel.stop_after_entry', True)
        self.declare_parameter('tunnel.give_up_s', 120.0)

        # Through the tunnel on Nav2, from the front of it (ARX). When the
        # robot stops at the front (stop_at_sign), it tells AMCL where that
        # is on the tunnel map - start_x/y/yaw, map frame - and sends Nav2 to
        # the exit in the same breath, while control_lane hands /cmd_vel over.
        #
        # The yaw: map yaw at the front is start_yaw_deg when the robot faces
        # start_odom_yaw_deg on odom, which is where the stop has always been
        # measured (-88.5). Off by a few degrees on a given run, it is off by
        # the same on the map, so AMCL starts from the heading the robot really
        # has rather than the one it had when the map was made.
        self.declare_parameter('tunnel.nav.enabled', True)
        self.declare_parameter('tunnel.nav.start_x', 0.0)
        self.declare_parameter('tunnel.nav.start_y', 0.0)
        self.declare_parameter('tunnel.nav.start_yaw_deg', 0.0)
        self.declare_parameter('tunnel.nav.start_odom_yaw_deg', -90.0)
        self.declare_parameter('tunnel.nav.goal_x', 1.85)
        self.declare_parameter('tunnel.nav.goal_y', 1.90)
        self.declare_parameter('tunnel.nav.goal_yaw_deg', 0.0)
        # Standard deviation of the pose given to AMCL - how far off the
        # stop may be from where the map says the front is.
        self.declare_parameter('tunnel.nav.start_sigma_m', 0.10)
        self.declare_parameter('tunnel.nav.start_sigma_deg', 10.0)
        # Wait for AMCL to take the pose before sending the goal: until it
        # reports the robot within localise_tol_m of start_x/y, sending the
        # pose again every half second, for at most localise_s. Just waiting
        # for the next /amcl_pose was not enough - AMCL runs all along from a
        # placeholder pose, and its next report was still where it had
        # drifted to (1.68, 1.27), so Nav2 planned from there and failed.
        self.declare_parameter('tunnel.nav.localise_s', 3.0)
        self.declare_parameter('tunnel.nav.localise_tol_m', 0.25)
        # Nav2 not answering in this long: it is not running. Stop.
        self.declare_parameter('tunnel.nav.server_wait_s', 5.0)
        self.declare_parameter('tunnel.nav.give_up_s', 60.0)
        # At the exit: follow the lane on from there (true) or stop. A failed
        # or timed-out Nav2 run stops either way - the robot is somewhere in
        # the tunnel, not on a lane.
        self.declare_parameter('tunnel.nav.follow_lane_after', True)

        self.declare_parameter('rate_hz', 10.0)

        # The world heading the robot is standing at before it moves (ARX).
        #
        # gz's DiffDrive integrates its odometry from zero whatever pose the
        # model was placed in, so /odom yaw is the angle turned since the
        # start rather than the angle in the world. Every heading.center_deg
        # here is a world angle, measured off the course, so the two have to
        # be tied together.
        #
        # It has been zero all along only because the example's spawn always
        # faces +x: spawn_turtlebot3.launch.py passes -x -y -z and no yaw at
        # all. A preset that stands the robot somewhere facing another way
        # has to say which way. Measured, not assumed - rotating the model
        # with gz's set_pose service from yaw -3.130 to 0.000 left /odom's
        # quaternion identical to twelve decimal places.
        self.declare_parameter('spawn_yaw_deg', 0.0)
        # Read once. It describes where the robot was put, not a knob to
        # turn, and a run cannot change it without respawning.
        self.spawn_yaw = float(self.get_parameter('spawn_yaw_deg').value)
        if self.spawn_yaw:
            self.get_logger().warn(
                f'odom yaw is offset by {self.spawn_yaw:+.1f} deg - the robot '
                f'was placed facing that way, and heading gates are in world '
                f'angles')

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
        # Skip the level crossing and go straight on to the tunnel (ARX). Set
        # from the command line like start_stage, not in mission.yaml, for the
        # same reason: a run that leaves a mission out is decided for that run.
        # level.enabled: false does the same thing from the file.
        self.declare_parameter('skip_level', False)

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
        self.yellow_ok = None
        self.white_ok = None
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
        # The left script (ARX): which step it is on, when that step began,
        # where the robot stood when a straight began, and whether it ran.
        self.left_step = 0
        self.left_step_since = None
        self.left_mark = None
        self.left_hold = None
        self.left_done = False
        # ARX: the sign arc: which side it is turning, done, and how many
        # ticks in a row with both lines.
        self.arc_side = None            # 'left' or 'right'
        self.arc_done = False
        self.arc_both = 0
        # ARX: right arc only - 0 aligning, 1 straight in, 2 the arc itself -
        # when that beat began, and where the straight began.
        self.arc_phase = 2
        self.arc_phase_since = None
        self.arc_mark = None
        # ARX: the white-line hug after the right arc - on, since when, and
        # the last heading, to see which way it is turning.
        self.right_hug = False
        self.right_hug_since = None
        self.right_hug_yaw = None
        # Latest report from detect_obstacle. None until one arrives, which
        # is not the same as a clear corridor and is treated differently.
        self.blocked = None
        # The line being hugged through the construction zone. White first:
        # the first board blocks the left half of the corridor.
        self.hug_side = FOLLOW_WHITE
        # Set while swinging round a board.
        self.edge_dir = 0
        # 0 arc out, 1 arc back, 2 straight, 3 arc out the other way,
        # 4 straight across, 5 arc back. Phases 2-5 only run with
        # double_swerve.
        self.edge_phase = 0
        self.edge_mark = None
        self.edge_mark_turn = 0.0
        # Boards got round since entering the zone. The first one is the
        # tight one, so it is the one that gets the slower, surer move.
        self.edge_count = 0
        # Set once a double swing has taken the robot round both boards.
        # The steering stays on the line it came back to - the parking sign
        # is what ends this stage now, not the boards - but the flag says
        # the threading part is behind us.
        self.zone_done = False
        self.corridor_yaw = None
        self.pos = None
        # The parking lot. Whether the yellow line has been solidly in view
        # in this stage yet, and where the robot was standing when it ran
        # out - which is the gap that is the pocket's mouth.
        self.lot_gone_ticks = 0
        # ARX: the parking sign, once confirmed, and since when both lines
        # have been in view without a break.
        self.parking_sign_at = None
        self.both_since = None
        self.white_lost_ticks = 0       # ARX: for construction.stop_when_white_lost
        self.yellow_lost_ticks = 0      # ARX: for parking.stop_when_yellow_lost
        self.parking_both_held = False
        self.park_arc_phase = None       # ARX: parking.left_arc - 0 straight, 1 arc
        # ARX: start_stage parking_17b - see START_PARKING_17B.
        self.parking_from_17b = self.stage == START_PARKING_17B
        if self.parking_from_17b:
            self.stage = STAGE_DRIVE_TO_PARKING
        self.lot_from_arc = False
        self.lot_pair_lost = 0
        self.wl_arc_since = None        # ARX: white_lost_arc running since
        self.wl_arc_back = 0
        # ARX: ticks in a row with both yellow walls, and whether that has
        # released the slow-down for the rest of the way in.
        self.lot_pair_ticks = 0
        self.lot_pair_seen = False
        self.lot_mark = None
        # The heading the corridor is being run on, once the robot has
        # settled onto one, and the window it is judged from.
        self.hold_heading_deg = None
        self.lot_recent = deque(maxlen=60)
        self.lot_turn = 0.0
        self.lot_last_yaw = None
        # Latest report from detect_obstacle's side wedges, and the bay
        # chosen from it.
        self.side = None
        self.side_at = None              # ARX: when self.side last came in
        self.bay = None
        # Latest report from detect_level, and how many ticks in a row it has
        # said the same thing.
        self.bar = None
        self.bar_range = None
        self.bar_range_at = None
        self.bar_ticks = 0
        # Ticks in a row the tunnel test has held, and whether the robot has
        # stopped steering by the lane and is driving itself in.
        self.tunnel_ticks = 0
        self.tunnel_going_in = False
        # The tunnel board has been confirmed; now waiting for the front.
        self.tunnel_sign_seen = False
        self.front_range = None
        # Backing into the bay: which phase, which way round, and the marks
        # each phase measures from.
        self.park_phase = 0
        self.park_way = 0
        self.park_mark = None
        self.park_turn = 0.0
        self.park_last_yaw = None
        self.park_since = None
        self.park_ticks = 0

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
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 1)
        # ARX: the tunnel on Nav2 - see tunnel.nav.
        self.pub_nav_active = self.create_publisher(Bool, '/arx/nav_active', 1)
        self.pub_initialpose = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 1)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl_pose, 1)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Where Nav2 thinks the robot is: map->base_footprint on TF, which is
        # what bt_navigator plans from. /amcl_pose is not - AMCL publishes the
        # new pose there the moment it takes /initialpose, but only moves the
        # map->odom TF on its next scan, and a goal sent in between was
        # planned from where AMCL had drifted to (1.43, 1.69) and failed.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Wiped once the pose has landed, before the goal: anything marked
        # while AMCL was on its placeholder pose is in the wrong place.
        self.clear_costmaps = [
            self.create_client(ClearEntireCostmap, name) for name in (
                '/global_costmap/clear_entirely_global_costmap',
                '/local_costmap/clear_entirely_local_costmap')]
        self.nav_want = False
        self.nav_phase = None
        self.nav_goal = None
        self.nav_result = None
        self.nav_pose_at = None
        self.amcl_pose_at = None
        self.nav_refused = False
        self.nav_sent_at = None
        self.amcl_xy = None
        self.create_subscription(UInt8, '/detect/lane_state', self.on_lane_state, 1)
        self.create_subscription(UInt8, '/arx/traffic_light', self.on_light, 1)
        self.create_subscription(UInt8, '/detect/traffic_sign', self.on_sign, 1)
        self.create_subscription(Odometry, '/odom', self.on_odom, 1)
        self.create_subscription(UInt8, '/detect/obstacle', self.on_obstacle, 1)
        self.create_subscription(UInt8, '/detect/obstacle_side', self.on_side, 1)
        self.create_subscription(Float32, '/detect/front_range', self.on_front, 1)
        self.create_subscription(UInt8, '/detect/level_bar', self.on_bar, 1)
        self.create_subscription(Float32, '/detect/level_range', self.on_bar_range, 1)
        # How much of the frame each line fills, as detect_lane sees it.
        # Both are computed every frame whatever follow_side says, so they
        # stay live while the robot is hugging one of them - which
        # lane_state does not: hugging white makes it 3 by definition, and
        # it can never report both lines while that is set.
        self.create_subscription(
            UInt8, '/detect/yellow_line_reliability', self.on_yellow_ok, 1)
        self.create_subscription(
            UInt8, '/detect/white_line_reliability', self.on_white_ok, 1)

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
        """Track the robot's heading, in degrees, -180 to 180, and where it is."""
        q = msg.pose.pose.orientation
        yaw = math.degrees(math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        # Back into world angles. Zero for every run that starts on the start
        # line, so this changes nothing that was measured before it - see
        # spawn_yaw_deg. Wrapped, because a window straddling +/-180 is the
        # normal case here rather than the odd one.
        self.yaw = (yaw + self.spawn_yaw + 180.0) % 360.0 - 180.0
        # Position is only ever used as a difference over a few tens of
        # centimetres, which is what odometry from wheels is good at: over a
        # whole course it came out 2.5 cm from where the simulator said the
        # robot was. It is in the odom frame, whose origin is wherever the
        # robot started, so the absolute numbers mean nothing.
        self.pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def on_obstacle(self, msg):
        """Which half of the corridor ahead detect_obstacle says is blocked."""
        self.blocked = msg.data

    def on_side(self, msg):
        """Which side of the robot detect_obstacle says is taken (ARX)."""
        self.side = msg.data
        self.side_at = self.now()

    def on_front(self, msg):
        """Nearest thing in the wide wedge ahead, in metres (ARX)."""
        self.front_range = msg.data

    def on_bar(self, msg):
        """Whether detect_level can see the crossing bar, and how (ARX)."""
        self.bar = msg.data

    def on_bar_range(self, msg):
        """How far detect_level says the bar is, in metres (ARX)."""
        self.bar_range = msg.data
        self.bar_range_at = self.now()

    def heading_ok(self, prefix=None):
        """Whether the robot is pointing a way this mission's board can be seen from."""
        prefix = prefix or self.sign_prefix()
        if not self.get_parameter(f'{prefix}.heading.enabled').value:
            return True
        if self.yaw is None:
            # No odometry yet. This used to accept, as the lesser evil - but
            # the only time it happens is the first moments after start, and
            # that is exactly when it did harm: run 1023_1044 confirmed "left"
            # 0.1 s in, at the start line facing 0 deg (the junction gate
            # wants 180 +/- 45), on five 4-inlier matches of an arm not yet
            # raised, and skipped the junction. /odom arrives within the
            # second; a robot that never gets it cannot do the missions
            # anyway, and the junction then times out to after_junction.
            self.get_logger().warn(
                'heading gate is on but no /odom yet - ignoring sign reports until it arrives',
                once=True)
            return False
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
                prefix = self.sign_prefix()
                want = self.get_parameter(f'{prefix}.heading.center_deg').value
                tol = self.get_parameter(f'{prefix}.heading.tolerance_deg').value
                self.get_logger().info(
                    f'facing {where} - sign reports ignored, '
                    f'want {want:+.0f}+/-{tol:.0f}')
        if not ok:
            self.sign_rejected += 1
            return
        self.sign_votes.append((self.now(), msg.data))

    def sign_prefix(self):
        """Which parameter block governs the sign hunt in this stage."""
        if self.stage == STAGE_DRIVE_TO_CONSTRUCTION:
            return 'construction'
        if self.stage == STAGE_DRIVE_TO_PARKING:
            return 'parking'
        if self.stage == STAGE_DRIVE_TO_TUNNEL:
            return 'tunnel'
        return 'intersection'

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

    def on_yellow_ok(self, msg):
        """How much of the frame the yellow line fills, 0 to 100."""
        self.yellow_ok = msg.data

    def on_white_ok(self, msg):
        """How much of the frame the white line fills, 0 to 100."""
        self.white_ok = msg.data

    def both_lines(self):
        """Whether detect_lane can see both lines right now (ARX).

        Read off the two reliability topics rather than lane_state, because
        lane_state says which line the steering is using: while follow_side
        pins it to white it is 3 by definition and will never be 2, however
        clearly the yellow line is in frame. The reliabilities are worked
        out from both masks every frame regardless.
        """
        need = self.get_parameter('parking.min_reliability').value
        if self.yellow_ok is None or self.white_ok is None:
            return False
        return self.yellow_ok >= need and self.white_ok >= need

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
        self.max_vel_want = None
        self.nav_want = False
        self.tick_stage(now)
        self.send_max_vel()
        # Every tick, like drive_enable: a control_lane restarted mid-run
        # picks it up, and any stage but tunnel_nav takes the wheel back.
        self.pub_nav_active.publish(Bool(data=self.nav_want))

    def sign_hunt_speed(self):
        """Ask for speed.sign_hunt on /control/max_vel, if set (ARX)."""
        slow = self.get_parameter('speed.sign_hunt').value
        if slow > 0:
            self.max_vel_want = slow

    def send_max_vel(self):
        """Publish the speed ceiling a stage asked for, or the normal one (ARX).

        Only once something has asked: until then control_lane keeps its own
        control.max_vel and nothing here touches it. After that it is sent on
        every change and once a second besides, because control_lane's
        subscription is depth 1 and volatile and a single message can be
        missed.
        """
        want = self.max_vel_want
        if want is None:
            if self.max_vel_sent is None:
                return
            want = self.get_parameter('control.max_vel').value
        self.max_vel_tick += 1
        if want != self.max_vel_sent or self.max_vel_tick >= 10:
            self.pub_max_vel.publish(Float64(data=float(want)))
            self.max_vel_sent, self.max_vel_tick = want, 0

    def tick_stage(self, now):

        if self.stage == STAGE_WAIT_GREEN:
            self.tick_wait_green(now)
        elif self.stage == STAGE_DRIVE_TO_SIGN:
            self.tick_drive_to_sign(now)
        elif self.stage == STAGE_TURN:
            self.tick_turn(now)
        elif self.stage == STAGE_ARC:
            self.tick_arc(now)
        elif self.stage == STAGE_LEFT_SCRIPT:
            self.tick_left_script(now)
        elif self.stage == STAGE_FOLLOW_SIDE:
            self.tick_follow_side()
        elif self.stage == STAGE_DRIVE_TO_CONSTRUCTION:
            self.tick_drive_to_construction(now)
        elif self.stage == STAGE_AVOID_CONSTRUCTION:
            self.tick_avoid_construction(now)
        elif self.stage == STAGE_EDGE_PAST_BOARD:
            self.tick_edge_past_board(now)
        elif self.stage == STAGE_DRIVE_TO_PARKING:
            self.tick_drive_to_parking(now)
        elif self.stage == STAGE_DRIVE_TO_LOT:
            self.tick_drive_to_lot(now)
        elif self.stage == STAGE_PICK_BAY:
            self.tick_pick_bay(now)
        elif self.stage == STAGE_PARK_IN_BAY:
            self.tick_park_in_bay(now)
        elif self.stage == STAGE_DRIVE_TO_LEVEL:
            self.tick_drive_to_level(now)
        elif self.stage == STAGE_APPROACH_BAR:
            self.tick_approach_bar(now)
        elif self.stage == STAGE_WAIT_BAR:
            self.tick_wait_bar(now)
        elif self.stage == STAGE_DRIVE_TO_TUNNEL:
            self.tick_drive_to_tunnel(now)
        elif self.stage == STAGE_TUNNEL_NAV:
            self.tick_tunnel_nav(now)
        else:
            self.tick_stopped()

    def tick_wait_green(self, now):
        self.set_armed(MISSION_TRAFFIC_LIGHT)
        self.set_driving(False)
        self.set_follow_side(FOLLOW_AUTO) #v;ldk
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
        self.sign_hunt_speed()
        self.set_follow_side(FOLLOW_AUTO) #kljvsldak
        self.set_avoid(False)

        if not self.get_parameter('intersection.enabled').value:
            self.set_armed(MISSION_NONE)
            return

        self.set_armed(MISSION_INTERSECTION)

        winner, votes = self.sign_majority()
        how = None
        if self.get_parameter('intersection.left_only.enabled').value:
            # ARX: see intersection.left_only.
            lefts = self.fresh_votes().count(SIGN_LEFT)
            right_at = self.get_parameter('intersection.left_only.right_at_deg').value
            winner = None
            if lefts >= self.get_parameter('intersection.left_only.confirm_votes').value:
                winner, votes = SIGN_LEFT, lefts
            elif self.heading_ok('intersection') and self.edge_reached(right_at, 1):
                winner = SIGN_RIGHT
                how = (f'no left arrow by {self.yaw:+.1f} deg (right_at {right_at:+.0f}, '
                       f'{lefts} left reports) - taking it as right')
        if winner is not None:
            self.decision = winner
            name = SIGN_NAMES[winner]
            self.get_logger().info(how or (
                f'sign {name} confirmed: {votes} of the last '
                f'{len(self.fresh_votes())} reports '
                f'({self.sign_msgs} reports seen, {self.sign_rejected} '
                f'ignored on heading)'))
            if self.get_parameter('intersection.stop_after_detect').value:
                self.stop(f'sign {name} - stopping here, as asked')
            elif (SIDE_FOR_SIGN.get(winner) == FOLLOW_YELLOW and
                    self.get_parameter('intersection.left_arc.enabled').value):
                self.arc_side, self.arc_both = 'left', 0
                self.go(STAGE_ARC, f'sign {name} - holding the left arc')
            elif (SIDE_FOR_SIGN.get(winner) == FOLLOW_WHITE and
                    self.get_parameter('intersection.right_arc.enabled').value):
                self.arc_side, self.arc_both = 'right', 0
                self.arc_phase = 0 if self.get_parameter(
                    'intersection.right_arc.align').value else 1
                self.arc_phase_since, self.arc_mark = now, self.pos
                self.go(STAGE_ARC, f'sign {name} - holding the right arc')
            elif (SIDE_FOR_SIGN.get(winner) == FOLLOW_YELLOW and
                    self.get_parameter('intersection.left_script.enabled').value):
                self.start_left_step(0, now)
                self.go(STAGE_LEFT_SCRIPT, f'sign {name} - driving the left script')
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

    # --- the left script (ARX) ----------------------------------------
    #
    #   0  turn on the spot to face left_script.heading_deg
    #   1  forward first_m, holding the heading it set off on
    #   2  turn turn_deg on the spot, relative
    #   3  forward second_m, holding the heading it set off on
    #
    # then steer by the yellow line.

    def start_left_step(self, step, now):
        self.left_step = step
        self.left_step_since = now
        self.left_mark = self.pos
        # The heading a straight holds; the turn measures from here too.
        self.left_hold = self.yaw
        self.turn_start_yaw = self.yaw
        self.turn_last_yaw = self.yaw
        self.turn_accum = 0.0

    def left_rate(self, remaining_deg):
        """Turn rate towards a remaining angle, eased over the last stretch."""
        p = 'intersection.left_script.'
        rate = self.get_parameter(p + 'rate').value
        slow = self.get_parameter(p + 'slow_within_deg').value
        if slow > 0 and abs(remaining_deg) < slow:
            rate *= max(abs(remaining_deg) / slow, 0.25)
        return rate if remaining_deg >= 0 else -rate

    def left_straight(self, metres):
        """Drive forward, steering back onto the heading the step began on.

        Returns True once `metres` have been covered.
        """
        gone = self.dist_from(self.left_mark)
        if gone is not None and gone >= metres:
            return True
        angular = 0.0
        if self.left_hold is not None and self.yaw is not None:
            err = (self.left_hold - self.yaw + 180.0) % 360.0 - 180.0
            limit = self.get_parameter('intersection.left_script.rate').value
            angular = max(-limit, min(limit, math.radians(err)))
        self.set_avoid(True, angular=angular,
                       linear=self.get_parameter('intersection.left_script.speed').value)
        return False

    def tick_left_script(self, now):
        p = 'intersection.left_script.'
        self.set_armed(MISSION_NONE)
        # detect_lane fits yellow throughout, so the hand-over is not cold.
        self.set_follow_side(FOLLOW_YELLOW)
        # True throughout, as in tick_turn: hold_tick would zero the wheels.
        self.set_driving(True)

        if self.yaw is None or self.pos is None:
            self.get_logger().warn(
                'no odometry, so the left script cannot be measured - '
                'going straight to the yellow line', once=True)
            self.finish_left_script('no odometry')
            return

        step = self.left_step
        done = False
        if step == 0:
            want = self.get_parameter(p + 'heading_deg').value
            err = (want - self.yaw + 180.0) % 360.0 - 180.0
            if abs(err) <= self.get_parameter(p + 'heading_tolerance_deg').value:
                done = True
            else:
                self.set_avoid(True, angular=self.left_rate(err), linear=0.0)
        elif step == 1:
            done = self.left_straight(self.get_parameter(p + 'first_m').value)
        elif step == 2:
            self.accumulate_turn()
            target = self.get_parameter(p + 'turn_deg').value
            remaining = target - self.turn_accum
            if (target >= 0 and remaining <= 0) or (target < 0 and remaining >= 0):
                done = True
            else:
                self.set_avoid(True, angular=self.left_rate(remaining), linear=0.0)
        elif step == 3:
            done = self.left_straight(self.get_parameter(p + 'second_m').value)
        else:
            self.finish_left_script('left script done')
            return

        waited = now - self.left_step_since
        if not done and waited > self.get_parameter(p + 'step_give_up_s').value:
            self.get_logger().warn(
                f'left script step {step} timed out after {waited:.0f} s - '
                f'moving on')
            done = True
        if done:
            # A zero between steps, so a turn does not bleed into a straight.
            self.set_avoid(True, angular=0.0, linear=0.0)
            self.get_logger().info(
                f'left script step {step} done, facing {self.yaw:+.1f} deg')
            self.start_left_step(step + 1, now)

    def tick_arc(self, now):
        """One fixed arc the arrow's way until both lines show, or time is up (ARX)."""
        p = f'intersection.{self.arc_side}_arc.'
        self.set_armed(MISSION_NONE)
        self.set_driving(True)
        # detect_lane on both lines throughout, so lane_state 2 means the
        # branch's own pair is in view, and the hand-over is not cold.
        self.set_follow_side(FOLLOW_AUTO)

        # ARX: the right arc's first beat - carry on round to the heading.
        if self.arc_side == 'right' and self.arc_phase == 0:
            want = self.get_parameter(p + 'align_heading_deg').value
            off = None if self.yaw is None else self.heading_off(want, self.yaw)
            took = now - self.arc_phase_since
            if (off is None or abs(off) <= self.get_parameter(p + 'align_tolerance_deg').value
                    or took >= self.get_parameter(p + 'align_timeout_s').value):
                self.get_logger().info(
                    f'right arc: aligned in {took:.1f} s, facing '
                    f'{"unknown" if self.yaw is None else f"{self.yaw:+.1f}"} deg '
                    f'(want {want:+.0f}) - straight in')
                self.arc_phase, self.arc_phase_since, self.arc_mark = 1, now, self.pos
            else:
                w = abs(self.get_parameter(p + 'align_rate').value)
                self.set_avoid(True, angular=w if off > 0 else -w,
                               linear=self.get_parameter(p + 'align_speed').value)
                return

        # ARX: the second beat - straight in, holding the aligned heading.
        if self.arc_side == 'right' and self.arc_phase == 1:
            want_m = self.get_parameter(p + 'straight_m').value
            gone = self.dist_from(self.arc_mark)
            took = now - self.arc_phase_since
            if (want_m <= 0 or gone is None or gone >= want_m
                    or took >= self.get_parameter(p + 'straight_timeout_s').value):
                self.get_logger().info(
                    f'right arc: straight {0.0 if gone is None else gone:.2f} m in '
                    f'{took:.1f} s - now turning right')
                self.arc_phase, self.arc_phase_since, self.arc_both = 2, now, 0
            else:
                w = 0.0
                if self.yaw is not None and self.get_parameter(p + 'align').value:
                    err = math.radians(self.heading_off(
                        self.get_parameter(p + 'align_heading_deg').value, self.yaw))
                    lim = abs(self.get_parameter(p + 'align_rate').value)
                    w = max(-lim, min(lim, 1.5 * err))
                self.set_avoid(True, angular=w,
                               linear=self.get_parameter(p + 'straight_speed').value)
                return

        rate = abs(self.get_parameter(p + 'rate').value)
        self.set_avoid(True, angular=rate if self.arc_side == 'left' else -rate,
                       linear=self.get_parameter(p + 'speed').value)

        # Timed from the arc's own start, not the align beat's.
        start = self.arc_phase_since if self.arc_side == 'right' and self.arc_phase_since else self.stage_since
        elapsed = now - start
        self.arc_both = self.arc_both + 1 if self.lane_state == 2 else 0
        need = self.get_parameter(p + 'both_ticks').value
        if (self.arc_both >= need
                and elapsed >= self.get_parameter(p + 'min_s').value):
            self.finish_arc(f'both lines for {need} ticks after {elapsed:.1f} s')
        elif (self.arc_side == 'right' and self.get_parameter(p + 'use_end_heading').value
                and self.edge_reached(self.get_parameter(p + 'end_heading_deg').value, -1)):
            self.finish_arc(f'reached {self.yaw:+.1f} deg after {elapsed:.1f} s')
        elif elapsed >= self.get_parameter(p + 'timeout_s').value:
            self.finish_arc(f'{elapsed:.1f} s up, lane_state {self.lane_state}')

    def finish_arc(self, why):
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        self.arc_done = True
        if self.arc_side == 'right' and self.get_parameter(
                'intersection.right_arc.hug_white').value:
            self.right_hug, self.right_hug_since, self.right_hug_yaw = True, self.now(), self.yaw
            self.after_junction(f'{why} - steering by the white line until '
                                f'{self.get_parameter("intersection.right_arc.hug_until_deg").value:+.0f} deg')
            return
        self.after_junction(f'{why} - following both lines from here')

    def watch_right_hug(self, now):
        """Let go of the white line once heading crosses hug_until_deg (ARX).

        Crossing, not "within a tolerance of": the sign of the difference to
        the target flips between one tick and the next, which a coarse tick
        cannot step over. And only while turning the release way, since the
        heading crosses the same value turning the other way first.
        """
        if not self.right_hug or self.yaw is None:
            return
        p = 'intersection.right_arc.'
        prev, self.right_hug_yaw = self.right_hug_yaw, self.yaw
        if now - self.right_hug_since > self.get_parameter(p + 'hug_timeout_s').value:
            self.right_hug = False
            self.get_logger().warn(
                f'white-line hug timed out facing {self.yaw:+.1f} deg - both lines')
            return
        if prev is None:
            return
        target = self.get_parameter(p + 'hug_until_deg').value
        a, b = self.heading_off(target, prev), self.heading_off(target, self.yaw)
        step = self.heading_off(self.yaw, prev)          # + is turning left
        want_left = self.get_parameter(p + 'hug_release_turn').value == 'left'
        turning_ok = step > 0 if want_left else step < 0
        if a * b <= 0 and abs(a) < 45 and abs(b) < 45 and turning_ok:
            self.right_hug = False
            self.get_logger().info(
                f'heading crossed {target:+.0f} deg turning '
                f'{"left" if want_left else "right"} after '
                f'{now - self.right_hug_since:.1f} s - back to both lines')

    def finish_left_script(self, why):
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        self.left_done = True
        if self.get_parameter('intersection.left_script.stop_after').value:
            self.stop(f'{why} - stopping here, as asked')
            return
        line = ('the yellow line' if self.get_parameter(
            'intersection.left_script.follow_yellow_after').value
            else 'both lines')
        self.after_junction(f'{why} - steering by {line} from here')

    def chosen_side(self):
        """The line to steer by, or auto when the arrow only picked a turn."""
        if self.construction_seen:
            return FOLLOW_AUTO
        # The sign arc hands over to both lines, whatever follow_chosen_line says.
        if self.right_hug:
            return FOLLOW_WHITE
        if self.arc_done:
            return FOLLOW_AUTO
        # The left script decides for itself, whatever follow_chosen_line says.
        if self.left_done:
            if self.get_parameter('intersection.left_script.follow_yellow_after').value:
                return FOLLOW_YELLOW
            return FOLLOW_AUTO
        if not self.get_parameter('intersection.follow_chosen_line').value:
            return FOLLOW_AUTO
        side = SIDE_FOR_SIGN.get(self.decision, FOLLOW_AUTO)
        if (side == FOLLOW_WHITE and
                not self.get_parameter('intersection.right_follow_white_after').value):
            return FOLLOW_AUTO
        return side

    def after_junction(self, why):
        """Where the wheel goes once the junction is behind us."""
        if self.get_parameter('construction.enabled').value:
            self.reset_votes()
            self.go(STAGE_DRIVE_TO_CONSTRUCTION, why)
        else:
            self.go(STAGE_FOLLOW_SIDE, why)

    def tick_follow_side(self):
        """Just drive. Nothing is being looked for."""
        self.watch_right_hug(self.now())
        self.set_armed(MISSION_NONE)
        self.set_driving(True)
        self.set_follow_side(self.chosen_side())
        self.set_avoid(False)

    def tick_white_lost_arc(self, now):
        """A fixed-radius arc while the white line is out of view (ARX)."""
        p = 'construction.white_lost_arc.'
        v = self.get_parameter(p + 'speed').value
        w = v / max(self.get_parameter(p + 'radius_m').value, 0.05)
        if self.get_parameter(p + 'side').value != 'left':
            w = -w
        self.set_avoid(True, angular=w, linear=v)
        elapsed = now - self.wl_arc_since
        # Both lines by default, not just white: white alone is still the
        # inner line only, with the yellow fallback that latched onto the board.
        until = self.get_parameter(p + 'until').value
        found = (1, 2) if until == 'yellow' else (2,)   # yellow_pair: 2 in pair mode
        self.wl_arc_back = self.wl_arc_back + 1 if self.lane_state in found else 0
        need = self.get_parameter(p + 'back_ticks').value
        why = None
        if self.wl_arc_back >= need and elapsed >= self.get_parameter(p + 'min_s').value:
            why = f'{until} for {need} ticks'
        elif 0 < self.get_parameter(p + 'timeout_s').value <= elapsed:
            why = f'{elapsed:.1f} s up, lane_state {self.lane_state}'
        if why is not None:
            self.wl_arc_since = None
            self.set_avoid(True, angular=0.0, linear=0.0)
            self.set_avoid(False)
            where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            self.get_logger().info(
                f'white-lost arc done after {elapsed:.1f} s ({why}), facing {where} '
                f'- following the lane again')

    def tick_drive_to_construction(self, now):
        """Carry on down the course until the construction sign is read."""
        self.watch_right_hug(now)
        self.set_driving(True)
        self.sign_hunt_speed()
        # ARX: set_avoid(False) comes after white_lost_arc below, not here.
        # Sent here, every tick of the arc published "off" and then "on",
        # control_lane dropped arc commands that landed in between and
        # slipped lane-following ones in - run 1014_2251, 7A: the arc's first
        # 0.4 s never reached the wheels and the robot kept turning left.
        # Still steering by whatever the junction chose. Going back to both
        # lines happens when the sign is read, not on the way to it.
        # ARX: white_lost_arc's 'yellow_pair' has detect_lane look for the
        # yellow pair for as long as the arc runs, so lane_state 2 means it.
        if (self.wl_arc_since is not None
                and self.get_parameter('construction.white_lost_arc.until').value == 'yellow_pair'):
            self.set_follow_side(FOLLOW_YELLOW_PAIR)
        else:
            self.set_follow_side(self.chosen_side())
        self.set_armed(MISSION_CONSTRUCTION)

        # ARX: see construction.white_lost_arc / stop_when_white_lost.
        if self.wl_arc_since is not None:
            self.tick_white_lost_arc(now)
            return
        # Right branch only: there the inner line is white and leaves the
        # view in 7A. On the left branch white is the outer line, and a right
        # arc there would be the wrong way.
        arc_on = (self.get_parameter('construction.white_lost_arc.enabled').value
                  and self.arc_side == 'right')
        if arc_on or self.get_parameter('construction.stop_when_white_lost').value:
            self.white_lost_ticks = (self.white_lost_ticks + 1
                                     if self.lane_state in (0, 1) else 0)
            need = self.get_parameter('construction.white_lost_ticks').value
            if self.white_lost_ticks >= need:
                where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                self.white_lost_ticks = 0
                if arc_on:
                    self.wl_arc_since, self.wl_arc_back = now, 0
                    if self.get_parameter('construction.white_lost_arc.until').value == 'yellow_pair':
                        self.set_follow_side(FOLLOW_YELLOW_PAIR)
                    self.get_logger().info(
                        f'white line gone for {need} ticks (lane_state {self.lane_state}, '
                        f'facing {where}) - arcing until '
                        f'{self.get_parameter("construction.white_lost_arc.until").value} is in view')
                    self.tick_white_lost_arc(now)
                    return
                self.stop(f'white line gone for {need} ticks (lane_state '
                          f'{self.lane_state}, facing {where}) - stopping here, as asked')
                return

        # Lane following has the wheel - the white_lost_arc above returns
        # before this while it runs, and hands back itself when it ends.
        self.set_avoid(False)

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
                self.edge_count = 0
                self.zone_done = False
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
        side = self.hug_side
        self.set_follow_side(side)

        if self.blocked is not None and self.blocked & BLOCKED_AHEAD:
            if self.get_parameter(
                    'construction.avoid.stop_at_obstacle').value:
                self.stop(f'board across the {SIDE_NAMES[side]} line - '
                          f'stopping here, as asked')
                return
            if self.get_parameter('construction.avoid.edge.enabled').value:
                self.start_edge(now)
                return

        if self.zone_done and self.get_parameter('parking.enabled').value:
            # Both boards are behind us. Nothing else in this zone is worth
            # looking for, so start hunting the next mission's sign.
            self.reset_votes()
            self.go(STAGE_DRIVE_TO_PARKING, 'boards done - on to the parking sign')
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

    def start_edge(self, now):
        """Begin swinging round a board, if there is anywhere to swing to."""
        # Away from the line being hugged. The board blocks the half the
        # robot is in, so the other half is where it has to go, and the line
        # it was steering by is what says which half that was.
        #
        # Not from the LiDAR, which was the first version and sent the robot
        # out of the corridor twice. Two things defeat it. The board reaches
        # across most of its own half, so from the white line the nearer end
        # of it sits in the left window and reads as "left blocked" while the
        # gap it leaves is further left than the window sees. And the edge of
        # the corridor is paint, so the right window past the white line
        # reads clear - open ground, as far as a laser is concerned.
        self.edge_dir = +1 if self.hug_side == FOLLOW_WHITE else -1
        if self.get_parameter('construction.avoid.edge.absolute').value:
            off = self.heading_off(
                self.get_parameter('construction.avoid.edge.swing_heading_deg').value,
                self.get_parameter('construction.avoid.edge.corridor_heading_deg').value)
            self.edge_dir = +1 if off > 0 else -1
        # The heading to come back to. Taken now, while lane following still
        # has the robot square to the corridor.
        self.corridor_yaw = self.yaw
        self.turn_start_yaw = self.yaw
        self.turn_last_yaw = self.yaw
        self.turn_accum = 0.0
        self.edge_phase = 0
        self.edge_mark = None
        self.edge_mark_turn = 0.0
        self.edge_count += 1
        way = 'left' if self.edge_dir > 0 else 'right'
        spot = self.edge_on_spot()
        self.go(STAGE_EDGE_PAST_BOARD,
                f'board {self.edge_count} across the '
                f'{SIDE_NAMES[self.hug_side]} line - going {way} round it '
                f'{"on the spot" if spot else "on an arc"}')

    def tick_edge_past_board(self, now):
        """Get round a board, and round the one after it.

        Seven phases. The first swing turns on the spot and then drives
        straight; every swing after it is an arc, which is quicker.

        Turning on the spot for the first one is not symmetry for its own
        sake. That board is the one the robot arrives at square and at speed
        with the least room to spare, and an arc spends the gap in front of
        it turning towards it - measured at 13.4 mm of clearance against
        35.3 mm on the spot. An arc also throws the outside of the robot
        wide, which in a corridor 0.485 m across put it 72 to 323 mm past the
        yellow line depending on the radius. Turning on the spot sweeps only
        the robot's own 0.112 m.

        linear.x is zero while turning on the spot, which is not the robot
        having stopped: 90 degrees at this rate takes under two seconds,
        against the 30 the rules allow before a run is ended.

        The swings are closed on yaw against the heading lane following had,
        which is the same fact as the white line lying flat in the camera
        rather than standing upright - the cue this was asked for.
        detect_lane cannot report it: its sliding windows walk up from the
        bottom of the frame fitting x as a function of y, and a flat line is
        not a function of y. It goes quiet instead, which looks exactly like
        losing the line for any other reason.
        """
        self.set_driving(True)
        self.set_armed(MISSION_CONSTRUCTION)
        self.accumulate_turn()

        g = self.get_parameter
        speed = g('construction.avoid.edge.speed').value
        swing = g('construction.avoid.edge.swing_deg').value
        double = g('construction.avoid.edge.double_swerve').value
        first_r = g('construction.avoid.edge.radius_m').value
        second_r = g('construction.avoid.edge.second_radius_m').value
        between = g('construction.avoid.edge.between_m').value
        across = g('construction.avoid.edge.cross_m').value
        out = g('construction.avoid.edge.out_m').value
        spot_rate = g('construction.avoid.edge.in_place_rate').value
        absolute = g('construction.avoid.edge.absolute').value and self.yaw is not None

        radius = first_r if self.edge_phase < 3 else second_r
        rate = speed / radius if radius > 0.0 else 0.0
        turned = self.turned_so_far() or 0.0
        # Phases 4-6 swing the other way, measured from where phase 3 ended.
        base = self.edge_mark_turn if self.edge_phase >= 4 else 0.0
        way = self.edge_dir if self.edge_phase < 3 else -self.edge_dir

        if self.edge_phase == 0:
            if self.edge_on_spot():
                w = spot_rate
                slow = g('construction.avoid.edge.slow_within_deg').value
                if absolute and slow > 0:
                    left = abs(self.heading_off(self.edge_headings()[0], self.yaw))
                    w *= min(1.0, max(left / slow, 0.25))
                self.set_avoid(True, angular=self.edge_dir * w, linear=0.0)
            else:
                self.set_avoid(True, angular=self.edge_dir * rate, linear=speed)
            if (self.edge_reached(self.edge_headings()[0], self.edge_dir) if absolute
                    else abs(turned) >= swing):
                self.edge_phase = 10 if self.three_beat() else 1
                self.edge_mark = self.pos
                self.get_logger().info(
                    f'turned {turned:+.0f} deg '
                    f'{"on the spot" if self.edge_on_spot() else "off the corridor"}'
                    + (' - three beats: arcing over the board' if self.edge_phase == 10
                       else f' - out {out:.2f} m before coming back'))
            return

        if self.edge_phase >= 10:
            self.tick_three_beat(now)
            return

        if self.edge_phase == 1:
            # Straight out, pointing across the corridor, before the arc
            # back. Skipped when the first swing was an arc, which has
            # already carried the robot across while it turned.
            if not self.edge_on_spot():
                self.edge_phase = 2
                return
            w = 0.0
            if absolute:
                # Hold swing_heading_deg, so an overshoot in the turn is not
                # driven out as a diagonal.
                err = math.radians(self.heading_off(self.edge_headings()[0], self.yaw))
                w = max(-spot_rate, min(spot_rate, 1.5 * err))
            self.set_avoid(True, angular=w, linear=speed)
            gone = self.edge_gone()
            if gone is None or gone >= out:
                self.edge_phase = 2
                self.get_logger().info(
                    f'out {0.0 if gone is None else gone:.2f} m - coming back')
            return

        if self.edge_phase == 2:
            self.set_avoid(True, angular=-self.edge_dir * rate, linear=speed)
            back = (self.edge_reached(self.edge_headings()[1], -self.edge_dir) if absolute
                    else self.edge_crossed(turned, 0.0, self.edge_dir))
            if not back:
                return
            if not double:
                self.finish_edge('round the board')
                return
            self.edge_phase = 3
            self.edge_mark = self.pos
            self.get_logger().info(
                f'past the board - running {between:.2f} m straight before '
                f'the next swing')
            return

        if self.edge_phase == 3:
            self.set_avoid(True, angular=0.0, linear=speed)
            gone = self.edge_gone()
            if gone is None or gone >= between:
                self.edge_phase = 4
                self.edge_mark_turn = turned
                self.get_logger().info(
                    f'ran {0.0 if gone is None else gone:.2f} m - swinging the '
                    f'other way now')
            return

        if self.edge_phase == 4:
            self.set_avoid(True, angular=way * rate, linear=speed)
            if (self.edge_reached(self.edge_headings()[2], way) if absolute
                    else abs(turned - base) >= swing):
                self.edge_phase = 5
                self.edge_mark = self.pos
                self.get_logger().info(
                    f'swung {turned - base:+.0f} deg the other way - '
                    f'crossing {across:.2f} m before coming back')
            return

        if self.edge_phase == 5:
            self.set_avoid(True, angular=0.0, linear=speed)
            gone = self.edge_gone()
            if gone is None or gone >= across:
                self.edge_phase = 6
                self.get_logger().info(
                    f'crossed {0.0 if gone is None else gone:.2f} m - coming back')
            return

        self.set_avoid(True, angular=-way * rate, linear=speed)
        if (self.edge_reached(self.edge_headings()[1], -way) if absolute
                else self.edge_crossed(turned, base, way)):
            self.finish_edge('round both boards')
            return

        waited = now - self.stage_since
        if waited > g('construction.avoid.edge.give_up_s').value:
            self.get_logger().warn(
                f'still getting round a board after {waited:.0f} s '
                f'(phase {self.edge_phase}, turned {turned:+.0f} deg) - '
                f'handing back to the lane')
            self.finish_edge('gave up getting round')

    def three_beat(self):
        """Whether the swerve after the spin is three_beat's two arcs (ARX)."""
        return (self.get_parameter('construction.avoid.edge.three_beat.enabled').value
                and self.get_parameter('construction.avoid.edge.absolute').value
                and self.yaw is not None)

    def tick_three_beat(self, now):
        """Phases 10 and 11: the two arcs after the spin (ARX).

        10 arcs back against the spin's way until arc1_end_deg - over the
        board just met and down the far side of it. 11 arcs the spin's way
        until the corridor heading. Phase 10 or more counts as both boards
        done in finish_edge, which is what the second arc is for.
        """
        g = self.get_parameter
        p = 'construction.avoid.edge.three_beat.'
        corridor = g('construction.avoid.edge.corridor_heading_deg').value
        wheel = g(p + 'wheel_max').value
        half = g(p + 'half_track').value
        if self.edge_phase == 10:
            r, way = g(p + 'arc1_radius_m').value, -self.edge_dir
            # arc1_end_deg is given for a swing to the left; mirrored when
            # the swing went right.
            target = corridor + self.edge_dir * (g(p + 'arc1_end_deg').value - corridor)
        else:
            r, way = g(p + 'arc2_radius_m').value, self.edge_dir
            target = corridor
        v = min(g(p + 'speed').value, wheel / (1.0 + half / max(r, 0.01)))
        w = v / max(r, 0.01)
        lead = math.degrees(w) / max(g('rate_hz').value, 0.1) * g(p + 'end_lead_ticks').value
        if self.heading_off(target, self.yaw) * way <= lead:
            took = now - self.stage_since
            if self.edge_phase == 10:
                self.edge_phase = 11
                self.get_logger().info(
                    f'over the board: facing {self.yaw:+.1f} deg after {took:.1f} s '
                    f'- arcing back to {corridor:+.0f}')
            else:
                self.finish_edge(f'round both boards in three beats, {took:.1f} s')
            return
        self.set_avoid(True, angular=way * w, linear=v)
        waited = now - self.stage_since
        if waited > g('construction.avoid.edge.give_up_s').value:
            self.get_logger().warn(
                f'still in the three-beat swerve after {waited:.0f} s '
                f'(phase {self.edge_phase}, facing {self.yaw:+.1f} deg) - '
                f'handing back to the lane')
            self.finish_edge('gave up getting round')

    @staticmethod
    def heading_off(target, yaw):
        """Degrees from yaw to target, wrapped to -180..180; + is left."""
        return (target - yaw + 180.0) % 360.0 - 180.0

    def edge_headings(self):
        """(out, back, other-way out) for the absolute swings (ARX)."""
        g = self.get_parameter
        corridor = g('construction.avoid.edge.corridor_heading_deg').value
        out = g('construction.avoid.edge.swing_heading_deg').value
        mirror = corridor - self.heading_off(out, corridor)
        return out, corridor, mirror

    def edge_reached(self, target, way):
        """Whether turning `way` has brought yaw to or past target (ARX).

        The same sign-change test as edge_crossed, so a coarse tick cannot
        step over it, but against a fixed odom heading.
        """
        return self.yaw is not None and self.heading_off(target, self.yaw) * way <= 0.0

    def edge_crossed(self, turned, base, way):
        """Whether a swing has come back past the heading it left (ARX).

        A sign change, not a tolerance. The swing went `way` from `base`, so
        it is back once the difference has no sign left in that direction -
        which a tick cannot step over, however coarse it is. The heading is
        then within one tick of the corridor, 1.9 to 3.4 degrees, and lane
        following takes over and finishes the job.
        """
        return (turned - base) * way <= 0.0

    def edge_on_spot(self):
        """Whether this swing turns on the spot rather than arcing (ARX)."""
        if not self.get_parameter('construction.avoid.edge.in_place').value:
            return False
        if self.get_parameter(
                'construction.avoid.edge.in_place_first_only').value:
            return self.edge_count <= 1
        return True

    def dist_from(self, mark):
        """Metres travelled since a mark was taken, or None (ARX)."""
        if mark is None or self.pos is None:
            return None
        return math.hypot(self.pos[0] - mark[0], self.pos[1] - mark[1])

    def edge_gone(self):
        """Metres travelled since the straight run began, or None (ARX)."""
        return self.dist_from(self.edge_mark)

    def finish_edge(self, why):
        """Let go of the wheel and say which line to steer by now."""
        # One zero command first, so control_lane is not left relaying the
        # last rotation it was given.
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        if self.edge_phase >= 3:
            # Two swings, so back on the side it started from.
            self.hug_side = (FOLLOW_WHITE if self.edge_dir > 0
                             else FOLLOW_YELLOW)
            self.zone_done = True
        else:
            self.hug_side = (FOLLOW_YELLOW if self.edge_dir > 0
                             else FOLLOW_WHITE)
        self.go(STAGE_AVOID_CONSTRUCTION,
                f'{why} - steering by the {SIDE_NAMES[self.hug_side]} line now')

    def tick_drive_to_parking(self, now):
        """Carry on down the course until the parking sign is read.

        Two conditions, not one. The heading gate is the same idea as the
        other two boards - this one is turned -90 degrees in the world file
        as well, so it can only be read along the world x axis.

        The second is that detect_lane has both lines. Coming out of the
        construction zone the robot is still in a corridor two lanes wide
        with boards in it, and a sign detector looking at that has plenty to
        match against by accident. Both lines in view is what says the road
        has gone back to an ordinary single lane, which is where the parking
        board actually is.
        """
        if self.park_arc_phase is not None:
            self.tick_parking_left_arc(now)
            return
        # ARX: started at parking_17b. Once both lines are actually in view
        # - so a robot stood where there is no yellow does not count that as
        # the yellow dropping out - take the sign as read and the hold as done.
        if self.parking_from_17b and self.both_lines():
            self.parking_from_17b = False
            self.parking_sign_at = now
            self.parking_both_held = True
            self.get_logger().info(
                'started at parking_17b: sign and both lines taken as done - '
                'watching for the yellow line to drop out')
        self.set_driving(True)
        self.sign_hunt_speed()
        self.set_avoid(False)
        # Keep steering by the line the threading left the robot on.
        #
        # The mean of both lines was tried and drove worse: the robot came
        # off the run along y 1.744 at x 1.28 and wandered south, with
        # lane_state 0 the whole way. Hugging white it held that line from
        # x 1.6 down to 0.66, which is the line the parking board stands on.
        self.set_follow_side(self.hug_side)
        self.set_armed(MISSION_PARKING)

        # ARX: the sign is latched, then both lines have to hold for
        # parking.both_lines_s before the steering goes over to yellow - the
        # moment the road is back to one ordinary lane with both lines in it.
        if self.parking_sign_at is None:
            winner, votes = self.sign_majority({SIGN_PARKING})
            if winner is not None:
                self.parking_sign_at = now
                self.get_logger().info(
                    f'parking sign confirmed: {votes} of the last '
                    f'{len(self.fresh_votes())} reports '
                    f'({self.sign_msgs} seen, {self.sign_rejected} ignored on '
                    f'heading) - waiting for both lines')
        if self.both_lines():
            if self.both_since is None:
                self.both_since = now
        else:
            self.both_since = None
        held = 0.0 if self.both_since is None else now - self.both_since
        need_both = self.get_parameter('parking.require_both_lines').value
        need_s = self.get_parameter('parking.both_lines_s').value
        # ARX: parking.stop_when_yellow_lost - after the sign and the both-lines
        # hold, carry on by white instead of turning in, and stop once yellow
        # then drops out.
        if self.get_parameter('parking.stop_when_yellow_lost').value:
            if (not self.parking_both_held and self.parking_sign_at is not None
                    and (not need_both or held >= need_s)):
                self.parking_both_held = True
                self.get_logger().info(
                    f'both lines for {held:.1f} s after the parking sign - '
                    f'watching for the yellow line to drop out')
            if self.parking_both_held:
                floor = self.get_parameter('parking.min_reliability').value
                gone = self.yellow_ok is None or self.yellow_ok < floor
                self.yellow_lost_ticks = self.yellow_lost_ticks + 1 if gone else 0
                need = self.get_parameter('parking.yellow_lost_ticks').value
                if self.yellow_lost_ticks >= need:
                    where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                    if self.get_parameter('parking.left_arc.enabled').value:
                        self.park_arc_phase, self.park_arc_since = 0, now
                        self.park_arc_mark, self.park_arc_yaw, self.park_arc_turn = \
                            self.pos, self.yaw, 0.0
                        self.get_logger().info(
                            f'yellow line gone for {need} ticks (yellow {self.yellow_ok}, '
                            f'facing {where}) - turning left into the lot at 17B')
                        self.tick_parking_left_arc(now)
                        return
                    self.stop(f'parking sign read {now - self.parking_sign_at:.1f} s ago, '
                              f'both lines held, then the yellow line went for {need} '
                              f'ticks (yellow {self.yellow_ok}, white {self.white_ok}, '
                              f'lane_state {self.lane_state}, facing {where}) - '
                              f'stopping here, as asked')
                    return
        elif self.parking_sign_at is not None and (not need_both or held >= need_s):
            self.get_logger().info(
                f'both lines for {held:.1f} s (yellow {self.yellow_ok}, white '
                f'{self.white_ok}) - over to the yellow line')
            if self.get_parameter('parking.stop_after_detect').value:
                self.stop('parking sign - stopping here, as asked')
            elif self.get_parameter('lot.enabled').value:
                self.go(STAGE_DRIVE_TO_LOT, 'parking sign read - on to the lot')
            else:
                self.go(STAGE_FOLLOW_SIDE, 'parking sign read')
            return

        waited = now - self.stage_since
        if waited > self.get_parameter('parking.give_up_s').value:
            recent = [SIGN_NAMES.get(v, v) for v in self.fresh_votes()]
            yaw = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            # Three different faults look the same from here, so name which:
            # no reports at all, reports that never formed a majority, or a
            # majority that was never allowed because the lines were not
            # both in view.
            if self.parking_sign_at is not None:
                why = (f'sign read {now - self.parking_sign_at:.0f} s ago but both '
                       f'lines never held {need_s:.1f} s (yellow {self.yellow_ok}, '
                       f'white {self.white_ok})')
                self.get_logger().warn(f'parking: {why} - carrying on')
                self.go(STAGE_FOLLOW_SIDE, 'gave up waiting for both lines')
                return
            self.get_logger().warn(
                f'no parking sign after {waited:.0f} s '
                f'({self.sign_msgs} reports, {self.sign_rejected} dropped on '
                f'heading, facing {yaw}, lines yellow {self.yellow_ok} '
                f'white {self.white_ok}, '
                f'last {len(recent)}: {recent}) - carrying on')
            self.go(STAGE_FOLLOW_SIDE, 'gave up on the parking sign')

    def tick_parking_left_arc(self, now):
        """Straight pre_m, then a left arc to end_heading_deg, then the lot (ARX)."""
        p = 'parking.left_arc.'
        g = self.get_parameter
        self.set_driving(True)
        self.set_armed(MISSION_NONE)
        self.set_follow_side(FOLLOW_AUTO)
        if self.yaw is not None and self.park_arc_yaw is not None:
            self.park_arc_turn += (self.yaw - self.park_arc_yaw + 180.0) % 360.0 - 180.0
        self.park_arc_yaw = self.yaw
        v = g(p + 'speed').value
        took = now - self.park_arc_since
        if self.park_arc_phase == 0:
            gone = self.dist_from(self.park_arc_mark)
            if gone is None or gone >= g(p + 'pre_m').value:
                self.park_arc_phase = 1
                self.get_logger().info(
                    f'parking arc: {0.0 if gone is None else gone:.2f} m straight - '
                    f'now left, R {g(p + "radius_m").value:.2f}')
            else:
                self.set_avoid(True, angular=0.0, linear=v)
                return
        w = v / max(g(p + 'radius_m').value, 0.05)
        target = g(p + 'end_heading_deg').value
        # Half a tick of turning early, as three_beat does.
        lead = math.degrees(w) / max(g('rate_hz').value, 0.1) * 0.5
        done = self.yaw is not None and self.heading_off(target, self.yaw) <= lead
        if done or took >= g(p + 'timeout_s').value:
            self.set_avoid(True, angular=0.0, linear=0.0)
            self.set_avoid(False)
            self.park_arc_phase = None
            # drive_to_lot then follows the yellow pair - no slow approach,
            # no waiting for the heading to settle - and locks the arc's own
            # heading once the pair is lost. See tick_drive_to_lot.
            self.lot_turn = self.park_arc_turn
            self.lot_from_arc = True
            self.lot_pair_lost = 0
            where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            why = (f'turned {self.park_arc_turn:+.0f} deg' if done
                   else f'{took:.1f} s up, turned {self.park_arc_turn:+.0f} deg')
            self.go(STAGE_DRIVE_TO_LOT,
                    f'parking arc done ({why}, facing {where}) - following the '
                    f'yellow pair into the lot')
            return
        self.set_avoid(True, angular=w, linear=v)

    def lot_gone(self):
        """Metres travelled since the mouth came into view, or None (ARX)."""
        return self.dist_from(self.lot_mark)

    def lot_watch_heading(self):
        """Whether the robot is now running straight down the corridor (ARX).

        Accumulated turn, not a difference of headings, for the reason given
        on turn_accum: the corridor heading is near -90 and the road before
        it near 180, and a difference wraps.
        """
        if self.yaw is None:
            return False
        if self.lot_last_yaw is not None:
            self.lot_turn += (self.yaw - self.lot_last_yaw + 180.0) % 360.0 - 180.0
        self.lot_last_yaw = self.yaw
        if abs(self.lot_turn) < self.get_parameter('lot.turn_in_deg').value:
            # Still on the road or still turning into the corridor. The
            # window is kept clear so the tick the turn finishes on cannot be
            # called steady on the strength of readings from before it.
            self.lot_recent.clear()
            return False

        self.lot_recent.append(self.yaw)
        need = self.get_parameter('lot.steady_ticks').value
        if len(self.lot_recent) < need:
            return False
        window = list(self.lot_recent)[-need:]
        if max(window) - min(window) > self.get_parameter('lot.steady_band_deg').value:
            return False
        self.hold_heading_deg = sum(window) / len(window)
        return True

    def hold_correction(self):
        """How hard to steer back onto the corridor heading (ARX)."""
        return self.correction_to(self.hold_heading_deg)

    def correction_to(self, target_deg):
        """How hard to steer onto target_deg, the lot's P term (ARX)."""
        if target_deg is None or self.yaw is None:
            return 0.0
        err = (target_deg - self.yaw + 180.0) % 360.0 - 180.0
        cap = self.get_parameter('lot.max_angular').value
        return max(-cap, min(cap, self.get_parameter('lot.kp').value * err))

    def tick_drive_to_lot(self, now):
        """Follow the yellow line into the parking lot, then a little further (ARX).

        Nothing has to be aimed at. The yellow line turns north off the road
        at x 0.631 and runs down both walls of the way in, so a robot steering
        by it drives itself through the mouth and down the middle of the
        corridor - measured at x 0.4855 against a corridor of x 0.381 to
        0.631, twice, agreeing to a millimetre. That is also why the steering
        changes to yellow here, which no stage before this one does.

        It ends where the yellow line does. At y 1.008 the walls stop being
        yellow and become the white dashes that divide the two bays, and
        detect_lane has nothing left to follow.

        lane_state is what says so, rather than the yellow reliability: it
        went to 0 at y 1.040 on both runs and was 1 at every sample before
        that, all the way from x 1.28. The reliability cannot be used for
        this - it collapses at x 1.0 out on the road, half a metre early,
        because the line slides out of the rows the mask counts rather than
        because the line stops.

        Steering is taken off lane following at the same moment. lane_state 0
        is exactly the case where detect_lane stops publishing /detect/lane
        at all, so control_lane would be left acting on whatever it was given
        last - which is what sent the robot wandering round the pocket when
        this stage was first run without an end.
        """
        self.set_driving(True)
        self.set_armed(MISSION_NONE)

        if self.hold_heading_deg is None and self.lot_from_arc:
            # ARX: in off parking.left_arc. The yellow pair until it is lost,
            # then the arc's heading. No slow approach, no settling.
            p = 'parking.left_arc.'
            self.set_follow_side(FOLLOW_YELLOW_PAIR)
            if self.lot_pair_seen:
                self.set_avoid(False)
            else:
                # Not the lane until the pair is really there: odom holds the
                # arc's end heading meanwhile. With one wall in view detect_lane
                # falls back to a single yellow, and out of the arc that once
                # took the right wall for the left line and turned the robot
                # from -90 to -114 in the pair_wait_s second (run 1017_2303).
                # A pair - two yellows a lane apart - is not fooled that way.
                target = self.get_parameter(p + 'end_heading_deg').value
                w = self.correction_to(target)
                v = min(self.get_parameter('lot.speed').value,
                        self.get_parameter('lot.wheel_max').value
                        - abs(w) * self.get_parameter('lot.half_track').value)
                self.set_avoid(True, angular=w, linear=max(v, 0.0))
            if self.lane_state == 2:
                self.lot_pair_ticks += 1
                self.lot_pair_lost = 0
                if (not self.lot_pair_seen and self.lot_pair_ticks
                        >= self.get_parameter('lot.pair_release_ticks').value):
                    self.lot_pair_seen = True
                    self.get_logger().info('both yellow walls in view - following them')
            else:
                self.lot_pair_ticks = 0
                self.lot_pair_lost += 1
            why = None
            if (self.lot_pair_seen and self.lot_pair_lost
                    >= self.get_parameter(p + 'pair_lost_ticks').value):
                why = (f'yellow pair lost for {self.lot_pair_lost} ticks '
                       f'(lane_state {self.lane_state})')
            elif (not self.lot_pair_seen and now - self.stage_since
                    >= self.get_parameter(p + 'pair_wait_s').value):
                why = 'no yellow pair after the arc'
            if why is not None:
                self.hold_heading_deg = self.get_parameter(p + 'end_heading_deg').value
                where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                self.get_logger().info(
                    f'{why}, facing {where} - holding {self.hold_heading_deg:+.0f} deg')
        elif self.hold_heading_deg is None:
            # Lane following, steering by the yellow line, all the way round
            # the turn and into the corridor.
            self.set_avoid(False)
            self.set_follow_side(
                FOLLOW_YELLOW_PAIR if self.get_parameter('lot.yellow_pair').value
                else FOLLOW_YELLOW)
            pairing = self.get_parameter('lot.yellow_pair').value
            if pairing and not self.lot_pair_seen:
                self.lot_pair_ticks = self.lot_pair_ticks + 1 if self.lane_state == 2 else 0
                if self.lot_pair_ticks >= self.get_parameter('lot.pair_release_ticks').value:
                    self.lot_pair_seen = True
                    self.get_logger().info(
                        'both yellow walls in view - back to full speed')
            slow = self.get_parameter('lot.approach_max_vel').value
            if (slow > 0 and not self.lot_pair_seen and abs(self.lot_turn)
                    >= self.get_parameter('lot.slow_from_deg').value):
                self.max_vel_want = slow
            if self.lot_watch_heading():
                self.get_logger().info(
                    f'straight down the corridor on {self.hold_heading_deg:+.1f} deg '
                    f'after turning {self.lot_turn:+.0f} - holding it from here')
        else:
            # Holding it. The corridor is straight and the yellow line is
            # about to end, so there is nothing lane following can add.
            w = self.hold_correction()
            v = min(self.get_parameter('lot.speed').value,
                    self.get_parameter('lot.wheel_max').value
                    - abs(w) * self.get_parameter('lot.half_track').value)
            self.set_avoid(True, angular=w, linear=max(v, 0.0))

        # Watched the whole time, whoever is steering: detect_lane keeps
        # working out lane_state regardless of who has the wheel.
        if self.lot_mark is None:
            # ARX: 3 counts as well as 0, once the corridor heading is held.
            # Steering by yellow (or the yellow pair), a frame with no yellow
            # falls through to detect_lane's ordinary logic - and in the lot
            # the white wall and dashes are in view, so it reports 3, white
            # only, not 0, and the end of the yellow line was never seen. 3
            # only after the hold: in the turn at 17B the inner yellow can
            # drop out for a moment with the road's white line in view.
            # With the yellow pair on, detect_lane reports 0 for any frame
            # without yellow - including a moment's loss of it in the turn -
            # so in that mode nothing counts until the corridor is held.
            held = self.hold_heading_deg is not None
            if self.get_parameter('lot.yellow_pair').value:
                # White (3) once both yellow walls have been seen means the
                # walls have ended and the lot's white has begun - no need to
                # wait for the heading hold. Before the pair, white is just
                # the road's edge in the turn.
                ran_out = ((self.lot_pair_seen and self.lane_state == 3) or
                           (held and self.lane_state in (0, 3)))
            else:
                ran_out = self.lane_state == 0 or (held and self.lane_state == 3)
            if ran_out:
                self.lot_gone_ticks += 1
                if self.lot_gone_ticks >= self.get_parameter('lot.confirm_ticks').value:
                    self.lot_mark = self.pos
                    # ARX: take the wheel here if the corridor never settled
                    # enough to hold. Lane following has no centre to give
                    # once the yellow walls end - the yellow pair reports the
                    # lot's white without steering by it - so control_lane
                    # would keep its last command: on 2026-10-12 that turned
                    # the robot from -84 to -47 deg over the 0.30 m lead, and
                    # parking from there went wrong all the way out. The
                    # heading now is the one the yellow walls lined it up on.
                    if self.hold_heading_deg is None and self.yaw is not None:
                        self.hold_heading_deg = self.yaw
                        self.get_logger().info(
                            f'corridor never held - holding {self.yaw:+.1f} deg '
                            f'through the lead')
                    where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                    self.get_logger().info(
                        f'yellow line has run out - the white dashes start '
                        f'here, facing {where}. Carrying on '
                        f'{self.get_parameter("lot.lead_m").value:.2f} m')
            else:
                # Consecutive, not cumulative: one stray frame is not the end
                # of the line.
                self.lot_gone_ticks = 0
        else:
            gone = self.lot_gone()
            if gone is not None and gone >= self.get_parameter('lot.lead_m').value:
                where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                self.get_logger().info(
                    f'in the parking lot after {gone:.2f} m, facing {where}')
                self.finish_lot('in the parking lot')
                return

        waited = now - self.stage_since
        if waited > self.get_parameter('lot.give_up_s').value:
            # Three different faults look the same from here: a robot that
            # never settled into the corridor, one that did and never reached
            # the end of the yellow line, and one that did and never covered
            # the lead.
            state = ('never settled into the corridor' if self.hold_heading_deg is None
                     else 'yellow line never ran out' if self.lot_mark is None
                     else f'{self.lot_gone():.2f} m of the lead done')
            self.get_logger().warn(
                f'not in the parking lot after {waited:.0f} s '
                f'(turned {self.lot_turn:+.0f} deg, lane_state '
                f'{self.lane_state}, {state}) - carrying on')
            self.finish_lot('gave up on the parking lot', stopped=False)

    def finish_lot(self, why, stopped=True):
        """Let go of the wheel, and stop or carry on (ARX)."""
        # One zero command first, so control_lane is not left relaying the
        # last creep it was given.
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        if not stopped:
            self.go(STAGE_FOLLOW_SIDE, why)
        elif self.get_parameter('lot.stop_at_end').value:
            self.stop(f'{why} - stopping here, as asked')
        elif self.get_parameter('bay.enabled').value:
            self.go(STAGE_PICK_BAY, f'{why} - reading the bays')
        else:
            self.go(STAGE_FOLLOW_SIDE, why)

    def tick_pick_bay(self, now):
        """Stand between the two bays and read which one is empty (ARX).

        Standing still, because the reading is a pair of distances to either
        side and the robot is 0.138 m wide in a column 0.25 m wide: rolling
        while it reads would change both of them.

        Everything here is the robot's own left and right. Which bay that is
        on the course depends on which way the robot came in, and it does not
        need to know - it turns towards the side that is empty.
        """
        self.set_driving(True)
        self.set_follow_side(FOLLOW_AUTO)
        # Still holding the wheel, at a standstill. Handing back to lane
        # following here would let it act on a frame with no lane in it.
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_armed(MISSION_PARKING)

        waited = now - self.stage_since
        if waited < self.get_parameter('bay.settle_s').value:
            return

        # ARX: a report from after the robot stopped, not one left over.
        # detect_obstacle is disarmed through drive_to_lot and only armed
        # again here, so the last side report can be from the road.
        if self.side_at is None or self.side_at < self.stage_since:
            if waited > self.get_parameter('bay.give_up_s').value:
                self.get_logger().warn(
                    'no report from detect_obstacle since stopping - it may not '
                    'be running. Taking the preferred bay unread')
                self.pick_bay(self.get_parameter('bay.prefer').value,
                              'nothing was read')
            return
        if self.side is None:
            if waited > self.get_parameter('bay.give_up_s').value:
                self.get_logger().warn(
                    'no report from detect_obstacle at all - it may not be '
                    'running. Taking the preferred bay unread')
                self.pick_bay(self.get_parameter('bay.prefer').value,
                              'nothing was read')
            return

        left_taken = bool(self.side & SIDE_LEFT)
        right_taken = bool(self.side & SIDE_RIGHT)
        prefer = self.get_parameter('bay.prefer').value
        seen = (f'left {"taken" if left_taken else "clear"}, '
                f'right {"taken" if right_taken else "clear"}')

        if left_taken and right_taken:
            # Not a reason to stand there: a run that never moves again ends
            # at 30 seconds by the rules, and a bay wrongly called taken
            # costs less than that.
            self.get_logger().warn(
                f'both bays read as taken ({seen}) - one of them is a bad '
                f'reading, taking the {prefer} as set by bay.prefer')
            self.pick_bay(prefer, seen)
        elif left_taken:
            self.pick_bay('right', seen)
        elif right_taken:
            self.pick_bay('left', seen)
        else:
            self.pick_bay(prefer, f'{seen}, taking the {prefer} as set by bay.prefer')

    def pick_bay(self, side, why):
        """Settle on a bay and say so (ARX)."""
        self.bay = side
        self.get_logger().info(
            f'bays read: {why} -> parking in the bay on the robot\'s {side}')
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        if self.get_parameter('bay.stop_after_pick').value:
            self.stop(f'{side} bay chosen - stopping here, as asked')
        elif self.get_parameter('bay.park.enabled').value:
            # Clockwise for the bay on the left, anticlockwise for the one on
            # the right, so that the bay ends up behind the robot. The exit
            # arc later turns the same way again, which is what brings it
            # back to the middle of the column facing out.
            self.park_way = -1 if side == 'left' else +1
            self.park_phase = 0
            self.park_turn = 0.0
            self.park_last_yaw = None
            self.park_mark = None
            self.park_since = None
            self.go(STAGE_PARK_IN_BAY, f'{side} bay chosen - backing in')
        else:
            self.go(STAGE_FOLLOW_SIDE, f'{side} bay chosen')

    def park_swept(self):
        """Degrees turned since this phase began, accumulated (ARX)."""
        if self.yaw is None:
            return 0.0
        if self.park_last_yaw is not None:
            self.park_turn += (self.yaw - self.park_last_yaw + 180.0) % 360.0 - 180.0
        self.park_last_yaw = self.yaw
        return abs(self.park_turn)

    def park_next(self, why, now):
        """Start the next phase with its counters cleared (ARX)."""
        self.park_phase += 1
        self.park_turn = 0.0
        self.park_last_yaw = None
        self.park_mark = self.pos
        self.park_since = now
        self.park_ticks = 0
        self.get_logger().info(f'park phase {self.park_phase} - {why}')

    def tick_park_in_bay(self, now):
        """Back into the empty bay, stand in it, and drive out again (ARX).

        Five phases, and the whole thing is one direction of turn:

            0 turn 90 on the spot, putting the empty bay behind the robot
            1 reverse into it
            2 stand there
            3 drive out on a quarter circle, turning the same way again
            4 hold that heading until there is a line to follow

        Reversing in rather than driving in, because the robot has to leave
        again. Nose-first it would have to reverse out blind down a column
        0.25 m wide; tail-first the whole exit is one forward arc, and the arc
        that leaves the bay is the arc that lines it up with the way out.

        The radius is not free. From the left bay's centre at (0.754, 0.754)
        facing -x, a quarter circle to the right on 0.248 m ends at (0.506,
        1.002) facing +y: the middle of the mouth, pointed up the corridor.
        The right bay is the mirror of it and takes the same radius.
        """
        self.set_driving(True)
        self.set_armed(MISSION_NONE)
        if self.park_since is None:
            self.park_since = now
            self.park_mark = self.pos

        way = self.park_way
        speed = self.get_parameter('bay.park.speed').value

        if self.park_phase == 0:
            # On the spot. The column is 0.25 m wide and the robot 0.225 m
            # across its diagonal, so there is no room to turn any other way.
            rate = self.get_parameter('bay.park.turn_rate').value
            swept = self.park_swept()
            if self.get_parameter('bay.park.absolute').value and self.yaw is not None:
                want = self.get_parameter(f'bay.park.face_{self.bay}_deg').value
                left = self.heading_off(want, self.yaw) * way
                slow = self.get_parameter('bay.park.slow_within_deg').value
                if slow > 0:
                    rate *= min(1.0, max(abs(left) / slow, 0.25))
                self.set_avoid(True, angular=way * rate, linear=0.0)
                # Turned past it, or turned the long way round by more than
                # half a turn - the heading has been crossed either way.
                there = left <= 0.0 or swept > 270.0
            else:
                self.set_avoid(True, angular=way * rate, linear=0.0)
                there = swept >= self.get_parameter('bay.park.turn_deg').value
            if there:
                self.park_next('reversing into the bay', now)

        elif self.park_phase == 1:
            back = self.get_parameter('bay.park.reverse_speed').value or speed
            self.set_avoid(True, angular=0.0, linear=-back)
            want = self.get_parameter(
                f'bay.park.into_{self.bay}_m').value
            want -= (self.get_parameter('bay.park.reverse_lead_ticks').value
                     * back / max(self.get_parameter('rate_hz').value, 0.1))
            gone = self.dist_from(self.park_mark)
            if gone is not None and gone >= want:
                self.park_next(f'parked, {gone:.3f} m in', now)

        elif self.park_phase == 2:
            self.set_avoid(True, angular=0.0, linear=0.0)
            if now - self.park_since >= self.get_parameter('bay.park.hold_s').value:
                self.park_next('driving out on the arc', now)

        elif self.park_phase == 3:
            radius = self.get_parameter('bay.park.radius_m').value
            out = self.get_parameter('bay.park.exit_speed').value or speed
            out = min(out, self.get_parameter('lot.wheel_max').value
                      / (1.0 + self.get_parameter('lot.half_track').value / max(radius, 0.01)))
            w = out / max(radius, 0.01)
            self.set_avoid(True, angular=way * w, linear=out)
            swept = self.park_swept()
            absolute = (self.get_parameter('bay.park.exit_absolute').value
                        and self.yaw is not None)
            if absolute:
                target = self.get_parameter(f'bay.park.exit_heading_{self.bay}_deg').value
                lead = math.degrees(w) / max(self.get_parameter('rate_hz').value, 0.1) * 0.5
                there = self.heading_off(target, self.yaw) * way <= lead or swept > 180.0
            else:
                there = swept >= self.get_parameter('bay.park.turn_deg').value
            if there:
                if self.get_parameter('bay.park.rejoin').value:
                    self.hold_heading_deg = (
                        self.get_parameter(f'bay.park.exit_heading_{self.bay}_deg').value
                        if absolute else self.yaw)
                    self.park_next('holding the way out until a line appears', now)
                else:
                    self.finish_park('out of the bay')
                    return

        elif self.park_phase == 4:
            # Same P term the approach used, on the heading the arc finished
            # on. There is no line here for another 50 mm or so.
            out = self.get_parameter('bay.park.exit_speed').value or speed
            w = self.hold_correction()
            out = min(out, self.get_parameter('lot.wheel_max').value
                      - abs(w) * self.get_parameter('lot.half_track').value)
            self.set_avoid(True, angular=w, linear=max(out, 0.0))
            if self.get_parameter('bay.park.exit_absolute').value:
                # The yellow line on the robot's left, not any yellow: the
                # lower bay's arc brings the far wall into view first. With
                # follow_side YELLOW detect_lane reports 1 only for a yellow
                # line in the left half of the frame (lane.yellow_left_half).
                self.set_follow_side(FOLLOW_YELLOW)
                seen = self.lane_state == 1
            else:
                seen = self.lane_state not in (None, 0)
            if seen:
                self.park_ticks += 1
                if self.park_ticks >= self.get_parameter('bay.park.rejoin_ticks').value:
                    self.park_next(
                        f'lane_state {self.lane_state}, yellow {self.yellow_ok} - '
                        f'following the yellow line out', now)
            else:
                self.park_ticks = 0

        else:
            # Lane following again, but named: the yellow line, because it is
            # the one that leads out of the pocket, and not the mean of both,
            # because there is no white line in here to take a mean with.
            self.set_avoid(False)
            self.set_follow_side(FOLLOW_YELLOW)
            want = self.get_parameter('bay.park.leave_heading_deg').value
            tol = self.get_parameter('bay.park.leave_tolerance_deg').value
            if self.yaw is not None:
                off = abs((self.yaw - want + 180.0) % 360.0 - 180.0)
                if off <= tol:
                    self.finish_park(
                        f'back on the road heading, {off:.0f} deg off {want:+.0f}')
                    return

        waited = now - self.stage_since
        if waited > self.get_parameter('bay.park.give_up_s').value:
            self.get_logger().warn(
                f'still in phase {self.park_phase} of parking after '
                f'{waited:.0f} s - giving up and carrying on')
            self.finish_park('gave up on the bay')

    def finish_park(self, why):
        """Let go of the wheel and go back to following the road (ARX)."""
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
        self.get_logger().info(f'parking done: {why}, facing {where}')
        # follow_side lands on FOLLOW_AUTO by itself here - chosen_side()
        # returns it whenever the junction never picked a line, which is the
        # case on any run that reaches the parking lot - so there is nothing
        # to set.
        # Always by way of drive_to_level, even with the crossing switched
        # off: that stage is where the switch is read, so a run started at
        # the mission5 preset skips it the same way a full run does.
        self.go(STAGE_DRIVE_TO_LEVEL, f'{why} - on to the level crossing')

    def bar_run(self, wanted):
        """How many ticks in a row the bar report has matched (ARX).

        `wanted` is a set, because the useful question at the crossing is not
        always one value: leaving is decided on "anything but down", since the
        detector alternates between up and nothing at close range and both of
        those mean the road is clear.
        """
        if self.bar in wanted:
            self.bar_ticks += 1
        else:
            self.bar_ticks = 0
        return self.bar_ticks

    def by_range(self):
        """Whether the crossing is decided from the range alone (ARX)."""
        return self.get_parameter('level.source').value == 'ultrasonic'

    def fresh_range(self):
        """The latest bar range, or None when there is none recent (ARX)."""
        if self.bar_range is None or self.bar_range_at is None:
            return None
        if self.now() - self.bar_range_at > self.get_parameter(
                'level.range_max_age_s').value:
            return None
        return self.bar_range

    def level_stop_m(self):
        """Where the creep stops: camera.stop_m or stop_m, by level.source (ARX)."""
        if self.by_range():
            return self.get_parameter('level.stop_m').value
        return self.get_parameter('level.camera.stop_m').value

    def range_run(self, test):
        """Ticks in a row a fresh range has passed `test` (ARX).

        The range-mode twin of bar_run, on the same counter. A tick with no
        fresh range breaks the run - silence is not evidence either way.
        """
        rng = self.fresh_range()
        if rng is not None and test(rng):
            self.bar_ticks += 1
        else:
            self.bar_ticks = 0
        return self.bar_ticks

    def tick_drive_to_level(self, now):
        """Carry on down the road until the crossing bar comes down (ARX).

        Nothing is hunted on the way. There is no sign for this mission - the
        example has a stop board at (-1.35, 1.04) and a detector for it, but
        the bar itself is the thing that has to be obeyed, and it is visible
        from 0.60 m which is further than the board is useful from.
        """
        if (not self.get_parameter('level.enabled').value
                or self.get_parameter('skip_level').value):
            # finish_level goes on to the tunnel, or back to following the
            # road if that is switched off too. Ending in follow_side here
            # used to mean a skipped crossing skipped the tunnel as well.
            self.finish_level('level crossing disabled')
            return

        self.set_driving(True)
        self.set_avoid(False)
        self.set_follow_side(self.chosen_side())
        self.set_armed(MISSION_LEVEL)

        need = self.get_parameter('level.confirm_ticks').value
        slow_m = self.get_parameter('level.slow_m').value
        cam_slow_m = self.get_parameter('level.camera.slow_m').value
        if not self.heading_ok('level'):
            # Pointing somewhere else - mid-curve, most likely. Whatever is
            # in front is not the bar, and the count starts again once the
            # robot is facing the road.
            self.bar_ticks = 0
        elif (self.range_run(lambda r: r < slow_m) if self.by_range()
              # Camera: DOWN, and near enough - camera.slow_m on its range.
              else self.range_run(lambda r: self.bar == BAR_DOWN and r <= cam_slow_m)
              ) >= need:
            self.hold_heading_deg = self.yaw
            self.bar_ticks = 0
            rng = 'unknown' if self.bar_range is None else f'{self.bar_range:.2f} m'
            where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            self.get_logger().info(
                f'crossing bar is down, {rng} away, facing {where} - creeping up to '
                f'{self.level_stop_m():.2f} m')
            self.lot_mark = self.pos
            self.go(STAGE_APPROACH_BAR, 'bar down')
            return

        waited = now - self.stage_since
        if waited > self.get_parameter('level.give_up_s').value:
            self.get_logger().warn(
                f'no crossing bar after {waited:.0f} s '
                f'(last report {BAR_NAMES.get(self.bar, self.bar)}) - carrying on')
            self.go(STAGE_FOLLOW_SIDE, 'gave up on the crossing')

    def tick_approach_bar(self, now):
        """Creep up to the bar and stop short of it (ARX).

        Driven from here rather than by lane following, so that the distance
        is the only thing deciding when to stop. The heading is the one the
        robot was holding when the bar was confirmed - the road is straight
        here, and there is no reason to let the lane follower steer while the
        robot is watching something else.
        """
        self.set_driving(True)
        self.set_armed(MISSION_LEVEL)
        if self.get_parameter('level.approach_follow_lane').value:
            # 2026-10-18: keep following the lane, only slower - the road up
            # to the bar is not always straight enough to hold one heading.
            self.set_avoid(False)
            self.set_follow_side(self.chosen_side())
            self.max_vel_want = self.get_parameter('level.approach_max_vel').value
        else:
            self.set_avoid(True, angular=self.hold_correction(),
                           linear=self.get_parameter('level.speed').value)

        gone = self.dist_from(self.lot_mark)
        rng = self.fresh_range() if self.by_range() else self.bar_range

        # The bar lifting while the robot is still creeping is the ordinary
        # case in the simulator, where the plugin opens it at 0.30 m. It is
        # not a fault, it is the crossing letting the robot through.
        # By range, lifted here means the range went back out past slow_m -
        # past clear_m would be true from the start, the creep begins near
        # 0.45 m.
        slow_m = self.get_parameter('level.slow_m').value
        if (self.range_run(lambda r: r > slow_m) if self.by_range()
                else self.bar_run({BAR_UP, BAR_NONE})) >= self.get_parameter(
                'level.clear_ticks').value:
            self.finish_level(f'bar lifted while creeping, {gone:.2f} m in')
            return

        if rng is not None and rng <= self.level_stop_m():
            self.get_logger().info(
                f'stopping {rng:.2f} m short of the bar, after {gone:.2f} m')
            self.set_avoid(True, angular=0.0, linear=0.0)
            self.bar_ticks = 0
            if self.get_parameter('level.stop_at_bar').value:
                self.stop('at the crossing bar - stopping here, as asked')
            else:
                self.go(STAGE_WAIT_BAR, 'at the bar')
            return

        if gone is not None and gone >= self.get_parameter('level.max_creep_m').value:
            # The range said to keep going for longer than the road has. Out
            # past about 0.50 m detect_level often resolves three of the four
            # bands and measures the span between the wrong two, so a reading
            # that never comes down is expected rather than surprising.
            self.get_logger().warn(
                f'crept {gone:.2f} m without the range reaching '
                f'{self.level_stop_m():.2f} m '
                f'(last {rng}) - standing here anyway')
            self.set_avoid(True, angular=0.0, linear=0.0)
            self.bar_ticks = 0
            self.go(STAGE_WAIT_BAR, 'ran out of creep')

    def tick_wait_bar(self, now):
        """Stand at the bar until it lifts (ARX)."""
        self.set_driving(True)
        self.set_armed(MISSION_LEVEL)
        self.set_avoid(True, angular=0.0, linear=0.0)

        need = self.get_parameter('level.clear_ticks').value
        if self.by_range():
            # Standing at stop_m, the bar post and the road ahead read well
            # past clear_m; the bar itself reads under it. Down/up by the Pi's
            # own threshold is not used - at 0.45 it took the post for the bar.
            clear_m = self.get_parameter('level.clear_m').value
            run = self.range_run(lambda r: r > clear_m)
        else:
            run = self.bar_run({BAR_UP, BAR_NONE})
        if run >= need:
            rng = self.fresh_range()
            self.finish_level(
                f'bar lifted after {now - self.stage_since:.0f} s'
                + (f', range {rng:.2f} m' if rng is not None else ''))
            return

        waited = now - self.stage_since
        if waited > self.get_parameter('level.wait_give_up_s').value:
            # Before thirty seconds, which is where the rules end a run for
            # standing still. A bar that will not lift is worth driving into
            # the mission after it rather than losing the whole run to.
            self.get_logger().warn(
                f'the bar has not lifted after {waited:.0f} s - going anyway, '
                f'before the thirty seconds that end a run')
            self.finish_level('gave up waiting for the bar')

    def finish_level(self, why):
        """Let go of the wheel and carry on down the course (ARX)."""
        self.set_avoid(True, angular=0.0, linear=0.0)
        self.set_avoid(False)
        self.set_armed(MISSION_NONE)
        self.get_logger().info(f'level crossing done: {why}')
        if self.get_parameter('tunnel.enabled').value:
            # The tunnel is a sign hunt now: start it with an empty box.
            self.reset_votes()
            self.tunnel_sign_seen = False
            self.tunnel_ticks = 0
            self.go(STAGE_DRIVE_TO_TUNNEL, f'{why} - on to the tunnel')
        else:
            self.go(STAGE_FOLLOW_SIDE, why)

    def tick_drive_to_tunnel(self, now):
        """Drive into the tunnel and stop once the wall is alongside (ARX).

        Two parts, because the lane runs out before the tunnel does.

            follow the lane, watching for the tunnel board
            the board, big enough, facing -90 +/- 30 -> hold that heading and
                go straight in
            walls alongside -> stop

        The front of the tunnel used to be read off the LiDAR - something
        inside 0.20 m ahead. On the real course it is the board in front of
        it, and only once it looks big (sign.tunnel.min_width_px), which is
        what says the robot has arrived rather than merely seen it.

        The walls come with the course. tunnel_wall is defined inline in
        turtlebot3_autorace_2020.world rather than through an include, which
        is why a search for a model of that name turns up nothing. What the
        course does not come with is a roof; full.launch.py builds one over
        the mouth so that the camera loses the lane inside, which is the
        condition this mission is about - and the reason the heading has to be
        taken while there is still a line to take it from.

        Being inside is still read off the LiDAR rather than an upward range
        finder. A sensor has to live inside the robot's model, that model
        belongs to the example, and gz's DetachableJoint has to be declared in
        the parent model as well, so adding one would mean forking the robot
        and both of its launch files.
        """
        self.set_driving(True)
        self.set_armed(MISSION_TUNNEL)

        # Checked in the second part as well: the heading is held through
        # it, so if it was wrong going in it is wrong coming to the wall too.
        facing = self.heading_ok('tunnel')

        if not self.tunnel_going_in:
            # Follow the lane until the tunnel board has been seen, big
            # enough and while facing the right way - detect_sign drops it
            # while it is small, on_sign while the heading is off. That is
            # the front of the tunnel.
            self.set_avoid(False)
            self.set_follow_side(self.chosen_side())
            where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            stop_at_front = self.get_parameter('tunnel.stop_at_sign').value

            if self.tunnel_sign_seen:
                # Second condition: keep following the lane, slowly, until
                # something is right in front.
                self.max_vel_want = self.get_parameter('tunnel.after_sign_max_vel').value
                near = self.get_parameter('tunnel.stop_ahead_m').value
                if self.front_range is not None and self.front_range <= near:
                    self.tunnel_ticks += 1
                    if self.tunnel_ticks >= self.get_parameter('tunnel.stop_ahead_ticks').value:
                        self.get_logger().info(
                            f'{self.front_range:.2f} m ahead after the tunnel sign - '
                            f'at the tunnel front, facing {where}')
                        if self.get_parameter('tunnel.nav.enabled').value:
                            self.nav_phase = None
                            self.go(STAGE_TUNNEL_NAV, 'at the tunnel front - Nav2 to the exit')
                        else:
                            self.stop('at the tunnel front - stopping here, as asked')
                        return
                else:
                    self.tunnel_ticks = 0
                winner = None
            else:
                winner, votes = self.sign_majority({SIGN_TUNNEL})

            if winner == SIGN_TUNNEL and stop_at_front:
                self.tunnel_sign_seen = True
                self.tunnel_ticks = 0
                self.get_logger().info(
                    f'tunnel sign, {votes} votes, facing {where} - driving on '
                    f'until something is within '
                    f'{self.get_parameter("tunnel.stop_ahead_m").value:.2f} m ahead')
            elif winner == SIGN_TUNNEL:
                self.tunnel_going_in = True
                self.tunnel_ticks = 0
                self.hold_heading_deg = self.yaw
                where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                self.get_logger().info(
                    f'tunnel sign, {votes} votes - at the tunnel front, '
                    f'holding {where} and going straight in')
        else:
            # No line left to steer by under the roof, so hold the heading the
            # lane gave and creep.
            self.set_avoid(True, angular=self.hold_correction(),
                           linear=self.get_parameter('tunnel.speed').value)
            if self.get_parameter('tunnel.need_both').value:
                walls = self.side == (SIDE_LEFT | SIDE_RIGHT)
                what = 'walls either side'
            else:
                walls = self.side is not None and bool(self.side & SIDE_RIGHT)
                what = 'wall alongside on the right'
            if walls and facing:
                self.tunnel_ticks += 1
                if self.tunnel_ticks >= self.get_parameter('tunnel.confirm_ticks').value:
                    where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                    self.get_logger().info(
                        f'{what} - in the tunnel, facing {where}')
                    self.set_avoid(True, angular=0.0, linear=0.0)
                    self.set_avoid(False)
                    if self.get_parameter('tunnel.stop_after_entry').value:
                        self.stop('in the tunnel - stopping here, as asked')
                    else:
                        self.go(STAGE_FOLLOW_SIDE, 'in the tunnel')
                    return
            else:
                self.tunnel_ticks = 0

        waited = now - self.stage_since
        if waited > self.get_parameter('tunnel.give_up_s').value:
            if self.tunnel_sign_seen:
                state = (f'saw the tunnel sign, but nothing came within '
                         f'{self.get_parameter("tunnel.stop_ahead_m").value:.2f} m ahead')
            elif not self.tunnel_going_in:
                state = 'never saw the tunnel sign'
            else:
                state = 'went in and never found the walls'
            yaw = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            self.get_logger().warn(
                f'not in the tunnel after {waited:.0f} s ({state}, '
                f'ahead {self.front_range}, side {self.side}, facing {yaw}, '
                f'{self.sign_msgs} sign reports, {self.sign_rejected} ignored '
                f'on heading) - carrying on')
            self.set_avoid(True, angular=0.0, linear=0.0)
            self.set_avoid(False)
            self.go(STAGE_FOLLOW_SIDE, 'gave up on the tunnel')

    def on_amcl_pose(self, msg):
        """Where AMCL last put the robot, so a goal waits for the pose to land (ARX)."""
        self.amcl_pose_at = self.now()
        self.amcl_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def map_xy(self):
        """The robot on the map as Nav2 sees it - latest map->base_footprint (ARX)."""
        try:
            t = self.tf_buffer.lookup_transform("map", "base_footprint", Time())
        except Exception:
            return None
        return (t.transform.translation.x, t.transform.translation.y)

    def send_initial_pose(self):
        """Tell AMCL the robot is at the front of the tunnel, on the map (ARX)."""
        g = self.get_parameter
        yaw = g('tunnel.nav.start_yaw_deg').value
        if self.yaw is not None:
            off = (self.yaw - g('tunnel.nav.start_odom_yaw_deg').value + 180.0) % 360.0 - 180.0
            yaw += off
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        # Stamp zero, not now: AMCL looks up base->odom at the stamp, and over
        # WiFi "now" is a few ms ahead of the newest TF - "Failed to transform
        # initial pose in time (extrapolation into the future)". Zero is the
        # latest it has.
        msg.pose.pose.position.x = float(g('tunnel.nav.start_x').value)
        msg.pose.pose.position.y = float(g('tunnel.nav.start_y').value)
        msg.pose.pose.orientation.z = math.sin(math.radians(yaw) / 2.0)
        msg.pose.pose.orientation.w = math.cos(math.radians(yaw) / 2.0)
        sxy = g('tunnel.nav.start_sigma_m').value ** 2
        syaw = math.radians(g('tunnel.nav.start_sigma_deg').value) ** 2
        cov = [0.0] * 36
        cov[0] = cov[7] = sxy
        cov[35] = syaw
        msg.pose.covariance = cov
        self.pub_initialpose.publish(msg)
        self.get_logger().info(
            f'initial pose to AMCL: ({msg.pose.pose.position.x:.2f}, '
            f'{msg.pose.pose.position.y:.2f}) facing {yaw:+.1f} deg on the map')

    def send_nav_goal(self):
        """Send Nav2 to the tunnel exit (ARX)."""
        g = self.get_parameter
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(g('tunnel.nav.goal_x').value)
        goal.pose.pose.position.y = float(g('tunnel.nav.goal_y').value)
        yaw = math.radians(g('tunnel.nav.goal_yaw_deg').value)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.nav_result = None
        self.nav_goal = None
        self.nav_refused = False
        self.nav_sent_at = self.now()
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.on_nav_goal)
        self.get_logger().info(
            f'Nav2 goal: the tunnel exit at ({goal.pose.pose.position.x:.2f}, '
            f'{goal.pose.pose.position.y:.2f})')

    def on_nav_goal(self, future):
        handle = future.result()
        if not handle.accepted:
            # bt_navigator rejects while its lifecycle is still coming up -
            # tick_tunnel_nav tries again until server_wait_s.
            self.nav_refused = True
            self.get_logger().warn('Nav2 refused the goal - bt_navigator not active yet?')
            return
        self.nav_goal = handle
        handle.get_result_async().add_done_callback(self.on_nav_result)

    def on_nav_result(self, future):
        self.nav_result = future.result().status

    def tick_tunnel_nav(self, now):
        """Nav2 from the front of the tunnel to its exit, then stop (ARX).

        The pose and the goal go out in the same moment the robot reaches the
        front - Nav2 and AMCL have been up since the start of the run, so
        there is nothing to wait for but AMCL taking the pose, about a scan.
        control_lane passes Nav2's commands to /cmd_vel while nav_want holds.
        """
        self.set_driving(False)
        self.set_avoid(False)
        self.set_armed(MISSION_NONE)
        self.nav_want = True
        g = self.get_parameter
        waited = now - self.stage_since

        if self.nav_phase is None:
            if not self.nav_client.server_is_ready():
                if waited > g('tunnel.nav.server_wait_s').value:
                    self.get_logger().warn(
                        f'no Nav2 after {waited:.0f} s - is full.launch running with '
                        f'nav:=true? Stopping at the tunnel front')
                    self.stop('no Nav2')
                return
            self.send_initial_pose()
            self.nav_pose_at = now
            self.nav_phase = 'localise'
            return

        if self.nav_phase == 'localise':
            sx, sy = g('tunnel.nav.start_x').value, g('tunnel.nav.start_y').value
            xy = self.map_xy()
            off = None if xy is None else math.hypot(xy[0] - sx, xy[1] - sy)
            if off is not None and off <= g('tunnel.nav.localise_tol_m').value:
                self.get_logger().info(
                    f'TF has the robot at ({xy[0]:.2f}, {xy[1]:.2f}) on the map, '
                    f'{off:.2f} m from the front - sending the goal')
            elif waited > g('tunnel.nav.localise_s').value:
                where = 'no map->base_footprint' if xy is None else f'({xy[0]:.2f}, {xy[1]:.2f})'
                self.get_logger().warn(
                    f'TF still has the robot at {where} after {waited:.1f} s '
                    f'- sending the goal anyway')
            else:
                if now - self.nav_pose_at > 0.5:
                    self.send_initial_pose()
                    self.nav_pose_at = now
                return
            for client in self.clear_costmaps:
                if client.service_is_ready():
                    client.call_async(ClearEntireCostmap.Request())
            self.send_nav_goal()
            self.nav_phase = 'going'
            return

        if self.nav_refused:
            if waited > g('tunnel.nav.server_wait_s').value:
                self.get_logger().warn(
                    f'Nav2 still refusing after {waited:.0f} s - its bringup probably '
                    f'aborted (see lifecycle_manager_navigation). Stopping at the tunnel front')
                self.stop('Nav2 not active')
            elif now - self.nav_sent_at > 0.5:
                self.send_nav_goal()
            return

        if self.nav_result is not None:
            if self.nav_result == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f'Nav2 reached the tunnel exit after {waited:.1f} s')
                if g('tunnel.nav.follow_lane_after').value:
                    # nav_want is not set on the next tick, so control_lane
                    # takes the wheel back and steers by the lane again.
                    self.go(STAGE_FOLLOW_SIDE, 'out of the tunnel - back on the lane')
                else:
                    self.stop('out of the tunnel')
            else:
                self.get_logger().warn(
                    f'Nav2 did not reach the exit (status {self.nav_result}) after {waited:.1f} s')
                self.stop('Nav2 failed in the tunnel')
            return

        if waited > g('tunnel.nav.give_up_s').value:
            self.get_logger().warn(f'Nav2 still going after {waited:.0f} s - cancelling')
            if self.nav_goal is not None:
                self.nav_goal.cancel_goal_async()
            self.stop('Nav2 timed out in the tunnel')

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

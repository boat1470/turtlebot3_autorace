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
from std_msgs.msg import Bool, Float32, UInt8

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

SIGN_NAMES = {SIGN_CONSTRUCTION: 'construction',
              SIGN_LEFT: 'left', SIGN_RIGHT: 'right',
              SIGN_PARKING: 'parking'}

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
MISSION_PARKING = 4
MISSION_LEVEL = 5
MISSION_TUNNEL = 6

STAGE_WAIT_GREEN = 'wait_green'
STAGE_DRIVE_TO_SIGN = 'drive_to_sign'
STAGE_TURN = 'turn'
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

# /detect/lane_state, from detect_lane: which line it is steering by.
LANE_STATE_FOR_SIDE = {FOLLOW_YELLOW: 1, FOLLOW_WHITE: 3}
# Both lines in view. detect_lane publishes 0 for neither.
LANE_STATE_BOTH = 2
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
    STAGE_EDGE_PAST_BOARD,
    STAGE_DRIVE_TO_PARKING,
    STAGE_DRIVE_TO_LOT,
    STAGE_PICK_BAY,
    STAGE_PARK_IN_BAY,
    STAGE_DRIVE_TO_LEVEL,
    STAGE_APPROACH_BAR,
    STAGE_WAIT_BAR,
    STAGE_DRIVE_TO_TUNNEL,
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
        # A cap on the creep, because /detect/level_range is only trustworthy
        # inside about 0.50 m: further out the detector often resolves three
        # of the four bands and the span it measures is between the wrong
        # two. The range is what aims; this is what stops a bad one running
        # the robot into the bar.
        self.declare_parameter('level.max_creep_m', 0.60)
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
        # The lane runs along y at this end of the course and the robot
        # arrives travelling -y, so -90.
        self.declare_parameter('tunnel.heading.enabled', True)
        self.declare_parameter('tunnel.heading.center_deg', -90.0)
        self.declare_parameter('tunnel.heading.tolerance_deg', 45.0)
        self.declare_parameter('tunnel.enter_m', 0.20)
        self.declare_parameter('tunnel.enter_ticks', 3)
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
        self.bay = None
        # Latest report from detect_level, and how many ticks in a row it has
        # said the same thing.
        self.bar = None
        self.bar_range = None
        self.bar_ticks = 0
        # Ticks in a row the tunnel test has held, and whether the robot has
        # stopped steering by the lane and is driving itself in.
        self.tunnel_ticks = 0
        self.tunnel_going_in = False
        self.tunnel_rejected = 0
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

    def on_front(self, msg):
        """Nearest thing in the wide wedge ahead, in metres (ARX)."""
        self.front_range = msg.data

    def on_bar(self, msg):
        """Whether detect_level can see the crossing bar, and how (ARX)."""
        self.bar = msg.data

    def on_bar_range(self, msg):
        """How far detect_level says the bar is, in metres (ARX)."""
        self.bar_range = msg.data

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
        if self.stage == STAGE_DRIVE_TO_CONSTRUCTION:
            return 'construction'
        if self.stage == STAGE_DRIVE_TO_PARKING:
            return 'parking'
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

        radius = first_r if self.edge_phase < 3 else second_r
        rate = speed / radius if radius > 0.0 else 0.0
        turned = self.turned_so_far() or 0.0
        # Phases 4-6 swing the other way, measured from where phase 3 ended.
        base = self.edge_mark_turn if self.edge_phase >= 4 else 0.0
        way = self.edge_dir if self.edge_phase < 3 else -self.edge_dir

        if self.edge_phase == 0:
            if self.edge_on_spot():
                self.set_avoid(True, angular=self.edge_dir * spot_rate, linear=0.0)
            else:
                self.set_avoid(True, angular=self.edge_dir * rate, linear=speed)
            if abs(turned) >= swing:
                self.edge_phase = 1
                self.edge_mark = self.pos
                self.get_logger().info(
                    f'turned {turned:+.0f} deg '
                    f'{"on the spot" if self.edge_on_spot() else "off the corridor"}'
                    f' - out {out:.2f} m before coming back')
            return

        if self.edge_phase == 1:
            # Straight out, pointing across the corridor, before the arc
            # back. Skipped when the first swing was an arc, which has
            # already carried the robot across while it turned.
            if not self.edge_on_spot():
                self.edge_phase = 2
                return
            self.set_avoid(True, angular=0.0, linear=speed)
            gone = self.edge_gone()
            if gone is None or gone >= out:
                self.edge_phase = 2
                self.get_logger().info(
                    f'out {0.0 if gone is None else gone:.2f} m - coming back')
            return

        if self.edge_phase == 2:
            self.set_avoid(True, angular=-self.edge_dir * rate, linear=speed)
            if not self.edge_crossed(turned, 0.0, self.edge_dir):
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
            if abs(turned - base) >= swing:
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
        if self.edge_crossed(turned, base, way):
            self.finish_edge('round both boards')
            return

        waited = now - self.stage_since
        if waited > g('construction.avoid.edge.give_up_s').value:
            self.get_logger().warn(
                f'still getting round a board after {waited:.0f} s '
                f'(phase {self.edge_phase}, turned {turned:+.0f} deg) - '
                f'handing back to the lane')
            self.finish_edge('gave up getting round')

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
        self.set_driving(True)
        self.set_avoid(False)
        # Keep steering by the line the threading left the robot on.
        #
        # The mean of both lines was tried and drove worse: the robot came
        # off the run along y 1.744 at x 1.28 and wandered south, with
        # lane_state 0 the whole way. Hugging white it held that line from
        # x 1.6 down to 0.66, which is the line the parking board stands on.
        self.set_follow_side(self.hug_side)
        self.set_armed(MISSION_PARKING)

        winner, votes = self.sign_majority({SIGN_PARKING})
        both = self.both_lines()
        need_both = self.get_parameter('parking.require_both_lines').value
        if winner is not None and (both or not need_both):
            self.get_logger().info(
                f'parking sign confirmed: {votes} of the last '
                f'{len(self.fresh_votes())} reports '
                f'({self.sign_msgs} seen, {self.sign_rejected} ignored on '
                f'heading, lines yellow {self.yellow_ok} white {self.white_ok})')
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
            self.get_logger().warn(
                f'no parking sign after {waited:.0f} s '
                f'({self.sign_msgs} reports, {self.sign_rejected} dropped on '
                f'heading, facing {yaw}, lines yellow {self.yellow_ok} '
                f'white {self.white_ok}, '
                f'last {len(recent)}: {recent}) - carrying on')
            self.go(STAGE_FOLLOW_SIDE, 'gave up on the parking sign')

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
        if self.hold_heading_deg is None or self.yaw is None:
            return 0.0
        err = (self.hold_heading_deg - self.yaw + 180.0) % 360.0 - 180.0
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

        if self.hold_heading_deg is None:
            # Lane following, steering by the yellow line, all the way round
            # the turn and into the corridor.
            self.set_avoid(False)
            self.set_follow_side(FOLLOW_YELLOW)
            if self.lot_watch_heading():
                self.get_logger().info(
                    f'straight down the corridor on {self.hold_heading_deg:+.1f} deg '
                    f'after turning {self.lot_turn:+.0f} - holding it from here')
        else:
            # Holding it. The corridor is straight and the yellow line is
            # about to end, so there is nothing lane following can add.
            self.set_avoid(True, angular=self.hold_correction(),
                           linear=self.get_parameter('lot.speed').value)

        # Watched the whole time, whoever is steering: detect_lane keeps
        # working out lane_state regardless of who has the wheel.
        if self.lot_mark is None:
            if self.lane_state == 0:
                self.lot_gone_ticks += 1
                if self.lot_gone_ticks >= self.get_parameter('lot.confirm_ticks').value:
                    self.lot_mark = self.pos
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
            self.set_avoid(True, angular=way * self.get_parameter('bay.park.turn_rate').value,
                           linear=0.0)
            if self.park_swept() >= self.get_parameter('bay.park.turn_deg').value:
                self.park_next('reversing into the bay', now)

        elif self.park_phase == 1:
            self.set_avoid(True, angular=0.0, linear=-speed)
            want = self.get_parameter(
                f'bay.park.into_{self.bay}_m').value
            gone = self.dist_from(self.park_mark)
            if gone is not None and gone >= want:
                self.park_next(f'parked, {gone:.3f} m in', now)

        elif self.park_phase == 2:
            self.set_avoid(True, angular=0.0, linear=0.0)
            if now - self.park_since >= self.get_parameter('bay.park.hold_s').value:
                self.park_next('driving out on the arc', now)

        elif self.park_phase == 3:
            radius = self.get_parameter('bay.park.radius_m').value
            self.set_avoid(True, angular=way * speed / radius, linear=speed)
            if self.park_swept() >= self.get_parameter('bay.park.turn_deg').value:
                if self.get_parameter('bay.park.rejoin').value:
                    self.hold_heading_deg = self.yaw
                    self.park_next('holding the way out until a line appears', now)
                else:
                    self.finish_park('out of the bay')
                    return

        elif self.park_phase == 4:
            # Same P term the approach used, on the heading the arc finished
            # on. There is no line here for another 50 mm or so.
            self.set_avoid(True, angular=self.hold_correction(), linear=speed)
            if self.lane_state not in (None, 0):
                self.park_ticks += 1
                if self.park_ticks >= self.get_parameter('bay.park.rejoin_ticks').value:
                    self.park_next(
                        f'lane_state {self.lane_state} - following the yellow '
                        f'line out', now)
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
        if self.get_parameter('level.enabled').value:
            self.go(STAGE_DRIVE_TO_LEVEL, f'{why} - on to the level crossing')
        else:
            self.go(STAGE_FOLLOW_SIDE, why)

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

    def tick_drive_to_level(self, now):
        """Carry on down the road until the crossing bar comes down (ARX).

        Nothing is hunted on the way. There is no sign for this mission - the
        example has a stop board at (-1.35, 1.04) and a detector for it, but
        the bar itself is the thing that has to be obeyed, and it is visible
        from 0.60 m which is further than the board is useful from.
        """
        self.set_driving(True)
        self.set_avoid(False)
        self.set_follow_side(self.chosen_side())
        self.set_armed(MISSION_LEVEL)

        need = self.get_parameter('level.confirm_ticks').value
        if self.bar_run({BAR_DOWN}) >= need:
            self.hold_heading_deg = self.yaw
            self.bar_ticks = 0
            rng = 'unknown' if self.bar_range is None else f'{self.bar_range:.2f} m'
            self.get_logger().info(
                f'crossing bar is down, {rng} away - creeping up to '
                f'{self.get_parameter("level.stop_m").value:.2f} m')
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
        self.set_avoid(True, angular=self.hold_correction(),
                       linear=self.get_parameter('level.speed').value)

        gone = self.dist_from(self.lot_mark)
        rng = self.bar_range

        # The bar lifting while the robot is still creeping is the ordinary
        # case in the simulator, where the plugin opens it at 0.30 m. It is
        # not a fault, it is the crossing letting the robot through.
        if self.bar_run({BAR_UP, BAR_NONE}) >= self.get_parameter(
                'level.clear_ticks').value:
            self.finish_level(f'bar lifted while creeping, {gone:.2f} m in')
            return

        if rng is not None and rng <= self.get_parameter('level.stop_m').value:
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
                f'{self.get_parameter("level.stop_m").value:.2f} m '
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
        if self.bar_run({BAR_UP, BAR_NONE}) >= need:
            self.finish_level(f'bar lifted after {now - self.stage_since:.0f} s')
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
            self.go(STAGE_DRIVE_TO_TUNNEL, f'{why} - on to the tunnel')
        else:
            self.go(STAGE_FOLLOW_SIDE, why)

    def tick_drive_to_tunnel(self, now):
        """Drive into the tunnel and stop once the wall is alongside (ARX).

        Two parts, because the lane runs out before the tunnel does.

            follow the lane, watching the wide wedge ahead
            something inside 0.20 m -> hold that heading and go straight in
            the right-hand wall alongside -> stop

        The walls come with the course. tunnel_wall is defined inline in
        turtlebot3_autorace_2020.world rather than through an include, which
        is why a search for a model of that name turns up nothing. What the
        course does not come with is a roof; full.launch.py builds one over
        the mouth so that the camera loses the lane inside, which is the
        condition this mission is about - and the reason the heading has to be
        taken while there is still a line to take it from.

        Entry is read off the LiDAR rather than an upward range finder. A
        sensor has to live inside the robot's model, that model belongs to the
        example, and gz's DetachableJoint has to be declared in the parent
        model as well, so adding one would mean forking the robot and both of
        its launch files. The LiDAR answers the same question with nothing
        forked, and without depending on the light.
        """
        self.set_driving(True)
        self.set_armed(MISSION_TUNNEL)

        # Nothing here counts unless the robot is pointing the way the mouth
        # can be entered from. Checked in both parts, not just the first: the
        # heading is held through the second, so if it was wrong going in it
        # is wrong coming to the wall as well.
        facing = self.heading_ok('tunnel')

        if not self.tunnel_going_in:
            self.set_avoid(False)
            self.set_follow_side(self.chosen_side())
            near = self.get_parameter('tunnel.enter_m').value
            close = self.front_range is not None and self.front_range <= near
            if close and not facing:
                self.tunnel_rejected += 1
                if self.tunnel_rejected == 1:
                    where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                    want = self.get_parameter('tunnel.heading.center_deg').value
                    tol = self.get_parameter('tunnel.heading.tolerance_deg').value
                    self.get_logger().info(
                        f'something {self.front_range:.2f} m ahead but facing '
                        f'{where}, want {want:+.0f}+/-{tol:.0f} - not the tunnel')
            if close and facing:
                self.tunnel_ticks += 1
                if self.tunnel_ticks >= self.get_parameter('tunnel.enter_ticks').value:
                    self.tunnel_going_in = True
                    self.tunnel_ticks = 0
                    self.hold_heading_deg = self.yaw
                    where = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
                    self.get_logger().info(
                        f'the tunnel mouth is {self.front_range:.2f} m ahead - '
                        f'holding {where} and going straight in')
            else:
                self.tunnel_ticks = 0
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
            state = ('never found the mouth' if not self.tunnel_going_in
                     else 'went in and never found the walls')
            yaw = 'unknown' if self.yaw is None else f'{self.yaw:+.1f} deg'
            self.get_logger().warn(
                f'not in the tunnel after {waited:.0f} s ({state}, '
                f'ahead {self.front_range}, side {self.side}, facing {yaw}, '
                f'{self.tunnel_rejected} ignored on heading) - carrying on')
            self.set_avoid(True, angular=0.0, linear=0.0)
            self.set_avoid(False)
            self.go(STAGE_FOLLOW_SIDE, 'gave up on the tunnel')

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

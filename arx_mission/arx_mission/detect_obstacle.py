#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
# Copyright 2026 boat1470 (changes marked # ARX)
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
# Author: ChanHyeong Lee
#
# ARX: convert_laserscan_to_points and the top-down drawing come from
# turtlebot3_autorace_mission/avoid_construction.py. Its four-state
# sidestep does not, for two measured reasons:
#
#   1. The state machine leaves AVOID_STRAIGHT when a Hough line search
#      finds one lane line of each slope. Run against real frames from this
#      course at five robot poses - centred, yawed 20 and 80 degrees, and
#      shifted 12 cm each way - it reported "no lane" every time, including
#      the pose where the robot sat squarely between both lines. Its
#      minLineLength of 100 rejects the ~105 px arcs Canny actually finds
#      here, because they curve. A state with no working exit and no timeout
#      drives forward at 0.03 m/s for as long as the run lasts.
#   2. It publishes /avoid_active and /avoid_control itself, which
#      mission_control already owns. Two publishers on those topics means
#      control_lane is handed a mixture with nothing to say which came from
#      where.
#
# So this node only reports what the LiDAR sees, and mission_control decides
# what to do about it - the same split as every other detector here.
#
# What it keeps from the fork:
#
#   1. convert_laserscan_to_points, unchanged apart from comments. It is
#      correct and short: the range mask also drops NaN and inf for free,
#      since any comparison against NaN is False.
#   2. The top-down picture, because it is genuinely the fastest way to see
#      why a threshold did or did not fire - but published as a topic
#      instead of shown with cv2.imshow. imshow was called from the LiDAR
#      callback, so on a Raspberry Pi 4 over SSH, with no display, it raises
#      inside the callback, rclpy.spin re-raises, and the node dies. Nothing
#      would then be avoiding anything.

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import UInt8

# ARX: must match mission_control.py.
MISSION_CONSTRUCTION = 3
MISSION_PARKING = 4
MISSION_TUNNEL = 6

# ARX: the published bitmask, relative to the robot rather than to the
# corridor. The first version named the bits after halves of the corridor
# and it was wrong in a way that only showed up on the course: a robot
# hugging the white line is already in the right half, so a board blocking
# that half is straight in front of it, not off to its right. Asking "is my
# side blocked" then tested a window 6 to 18 cm to the side of the robot,
# which is where the neighbouring half is - and the run stopped against the
# construction sign standing beside the track instead.
#
#   AHEAD  the strip the robot itself occupies. Cannot carry on.
#   LEFT   the strip one width to its left. Nowhere to swap to.
#   RIGHT  the strip one width to its right.
BLOCKED_NONE = 0
BLOCKED_AHEAD = 1
BLOCKED_LEFT = 2
BLOCKED_RIGHT = 4

# ARX: the second report, for the parking lot, on its own topic. Same two
# names and the same meaning - the robot's own left and right - but measured
# quite differently, and the difference matters:
#
#   /detect/obstacle       boxes in FRONT of the robot, for driving into.
#                          Asks "can I carry on, and is there anywhere to
#                          swap to".
#   /detect/obstacle_side  wedges BESIDE the robot, for standing between two
#                          parking bays and asking which one is empty.
#
# Wedges rather than boxes here because the robot is standing still and
# looking sideways at something 0.2 to 0.3 m away: a box would have to be
# tuned for how deep into the bay the other robot happens to be parked, while
# a wedge only asks whether anything is over there at all. It is also what
# the 2018 example did - turtlebot3_autorace_detect/detect_parking.py scans
# beams 60 to 120 and 240 to 300 against 0.5 m - and that part of it is
# sound, unlike its dead-reckoned approach.
#
# The tunnel uses the same report for a different question. Driving up to it
# the sides are empty; from the mouth inwards there is wall on both, and the
# near one sits at a flat 0.120 m over sixty beams. Measured:
#
#   outside, y +0.60    left 1.081   right nothing
#   outside, y +0.20    left 0.673   right nothing
#   the mouth, y -0.10  left 0.169   right 0.120
#   inside,  y -0.35    left 0.355   right 0.120
#
# Which is why there is no upward range finder on this robot for the tunnel:
# a sensor has to live inside the model, that model belongs to the example,
# and gz's DetachableJoint has to be declared in the parent model too - so
# adding one would mean forking the robot and both of its launch files. The
# LiDAR already answers the question, and answers it without depending on
# the light.
SIDE_CLEAR = 0
SIDE_LEFT = 1
SIDE_RIGHT = 2

# ARX: the third report, on /detect/front_range: the nearest thing in a wide
# wedge straight ahead, in metres. Published as a distance rather than a bit
# because what wants it is a decision about when to stop steering by the lane
# and go straight - and that is a question about how far, not whether.
#
# Wide, at 45 degrees either side, because the thing it has to catch is the
# mouth of the tunnel, and the walls that mark it are off to both sides of
# the gap rather than square in front.
RANGE_NONE = 9.9


class DetectObstacle(Node):

    def __init__(self):
        super().__init__('detect_obstacle')

        # ARX: how far ahead counts as blocking, and how wide a corridor to
        # look down. See obstacle.yaml for why half_width is narrower than
        # the robot.
        self.declare_parameter('obstacle.stop_distance', 0.25)
        self.declare_parameter('obstacle.half_width', 0.06)
        self.declare_parameter('obstacle.min_points', 3)
        # ARX: the side wedges. 30 degrees either side of straight left and
        # straight right, out to half a metre - the example's numbers, and
        # they hold here: standing in the middle column with a robot in a bay
        # the nearest return measured 0.216 m against 1.821 m on the empty
        # side, so there is nearly a factor of ten between the two answers
        # and the threshold has plenty of room.
        self.declare_parameter('obstacle.side_angle_deg', 30.0)
        self.declare_parameter('obstacle.side_distance', 0.5)
        # More than one beam, unlike the example, which acted on the first
        # return under the threshold. At 1 degree per beam a bay's worth of
        # robot fills about 24 of them, so asking for 3 costs nothing and
        # keeps a single stray return from choosing the bay.
        self.declare_parameter('obstacle.side_min_points', 3)
        self.declare_parameter('obstacle.front_angle_deg', 45.0)
        self.declare_parameter('obstacle.frame_skip', 1)
        self.declare_parameter('obstacle.publish_debug_image', True)
        self.declare_parameter('obstacle.always_on', False)

        self.armed = 0
        self.counter = 1
        self.blocked = None
        self.side = None
        self.front = None

        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        # ARX: the gate every detector in this package has. A scan costs far
        # less than a SIFT frame, but the rules allow props on the course
        # that belong to no mission, and a detector left looking will find
        # one.
        self.create_subscription(UInt8, '/arx/armed_mission', self.cbArmedMission, 1)

        self.pub_obstacle = self.create_publisher(UInt8, '/detect/obstacle', 10)
        self.pub_side = self.create_publisher(UInt8, '/detect/obstacle_side', 10)
        self.pub_front = self.create_publisher(Float32, '/detect/front_range', 10)
        self.pub_image = self.create_publisher(
            CompressedImage, '/detect/image_output/compressed', 10)

        self.bridge = CvBridge()
        self.get_logger().info('DetectObstacle Node Initialized')

    def cbArmedMission(self, msg):
        """Remember which mission the sequencer says is running (ARX)."""
        if msg.data == self.armed:
            return
        self.armed = msg.data
        self.get_logger().info(f'armed mission -> {self.armed}')

    def fnArmed(self):
        """Whether this detector should be looking at all (ARX)."""
        if self.get_parameter('obstacle.always_on').value:
            return True
        return self.armed in (MISSION_CONSTRUCTION, MISSION_PARKING,
                              MISSION_TUNNEL)

    def lidar_callback(self, msg):
        # ARX: with nothing to look for, no work is done at all.
        if not self.fnArmed():
            return

        frame_skip = self.get_parameter('obstacle.frame_skip').value
        if frame_skip > 1:
            if self.counter % frame_skip != 0:
                self.counter += 1
                return
            self.counter = 1

        points = self.convert_laserscan_to_points(msg)
        blocked = self.fnBlocked(points)
        side = self.fnBlockedSide(msg)
        front = self.fnFrontRange(msg)

        # ARX: published every scan, not only on a change. mission_control
        # reads the latest value each tick, and a node that comes up late
        # would otherwise wait for the world to change before hearing
        # anything.
        out = UInt8()
        out.data = blocked
        self.pub_obstacle.publish(out)

        out_side = UInt8()
        out_side.data = side
        self.pub_side.publish(out_side)

        out_front = Float32()
        out_front.data = front
        self.pub_front.publish(out_front)

        # Logged in 5 cm steps, not on every change: it is a distance from a
        # moving robot, so it changes on every scan and a log of that is
        # unreadable.
        step = None if front >= RANGE_NONE else round(front / 0.05)
        if step != self.front:
            self.front = step
            if step is not None:
                self.get_logger().info(f'ahead -> {front:.3f} m')

        if blocked != self.blocked:
            self.blocked = blocked
            self.get_logger().info(f'blocked -> {self.fnBlockedName(blocked)}')

        if side != self.side:
            self.side = side
            near = self.fnSideRanges(msg)
            self.get_logger().info(
                f'beside -> {self.fnSideName(side)} '
                f'(nearest left {near[0]}, right {near[1]})')

        if self.get_parameter('obstacle.publish_debug_image').value:
            self.fnPublishImage(points, blocked)

    def convert_laserscan_to_points(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        # The mask drops NaN and inf as well as out-of-range readings,
        # because any comparison against NaN is False.
        valid = (ranges >= msg.range_min) & (ranges <= msg.range_max)
        ranges = ranges[valid]
        angles = angles[valid]
        if len(ranges) == 0:
            return np.array([])
        # Not REP-103: y is straight ahead and x is sideways, positive to
        # the right. The danger-zone test below reads as "in front of me and
        # not off to one side" because of it.
        x = ranges * -np.sin(angles)
        y = ranges * np.cos(angles)
        return np.vstack((x, y)).T

    def fnBlocked(self, points):
        """Which of the three strips ahead of the robot has something in it (ARX).

        Three windows the same width, side by side: the one the robot is
        driving down, and one to each side of it. Only the middle one says
        whether it can carry on; the outer two say whether there is anywhere
        to go instead.
        """
        if points is None or len(points) == 0:
            return BLOCKED_NONE

        reach = self.get_parameter('obstacle.stop_distance').value
        half = self.get_parameter('obstacle.half_width').value
        need = self.get_parameter('obstacle.min_points').value

        x, y = points[:, 0], points[:, 1]
        ahead = (y > 0.0) & (y < reach)

        blocked = BLOCKED_NONE
        if np.count_nonzero(ahead & (np.abs(x) <= half)) >= need:
            blocked |= BLOCKED_AHEAD
        if np.count_nonzero(ahead & (x < -half) & (x > -3.0 * half)) >= need:
            blocked |= BLOCKED_LEFT
        if np.count_nonzero(ahead & (x > half) & (x < 3.0 * half)) >= need:
            blocked |= BLOCKED_RIGHT
        return blocked

    def fnBlockedSide(self, msg):
        """Which side of the robot has something standing next to it (ARX).

        Straight off the scan rather than through convert_laserscan_to_points,
        because the question is angular: the beam at 90 degrees is the robot's
        left however the robot is turned, while an x/y window would have to be
        rotated to mean the same thing.
        """
        n = len(msg.ranges)
        if n == 0:
            return SIDE_CLEAR

        half = self.get_parameter('obstacle.side_angle_deg').value
        reach = self.get_parameter('obstacle.side_distance').value
        need = self.get_parameter('obstacle.side_min_points').value

        angles = np.linspace(msg.angle_min, msg.angle_max, n)
        ranges = np.array(msg.ranges)
        # Drops NaN and inf as well, because any comparison against NaN is
        # False.
        near = ((ranges >= msg.range_min) & (ranges <= msg.range_max)
                & (ranges <= reach))
        # Wrapped, because a scan that runs 0 to 360 puts the robot's right
        # at 270 while one that runs -180 to 180 puts it at -90.
        deg = (np.degrees(angles) + 180.0) % 360.0 - 180.0

        side = SIDE_CLEAR
        if np.count_nonzero(near & (np.abs(deg - 90.0) <= half)) >= need:
            side |= SIDE_LEFT
        if np.count_nonzero(near & (np.abs(deg + 90.0) <= half)) >= need:
            side |= SIDE_RIGHT
        return side

    def fnFrontRange(self, msg):
        """Nearest return in a wide wedge straight ahead, in metres (ARX).

        Wide on purpose. The narrow box on /detect/obstacle asks "can I carry
        on", which is the construction zone's question; this one has to notice
        the mouth of a corridor, whose walls stand off to both sides of the
        gap the robot is aiming at rather than square in front of it.
        """
        n = len(msg.ranges)
        if n == 0:
            return RANGE_NONE
        half = self.get_parameter('obstacle.front_angle_deg').value
        angles = np.linspace(msg.angle_min, msg.angle_max, n)
        ranges = np.array(msg.ranges)
        ok = (ranges >= msg.range_min) & (ranges <= msg.range_max)
        deg = (np.degrees(angles) + 180.0) % 360.0 - 180.0
        sel = ok & (np.abs(deg) <= half)
        return float(ranges[sel].min()) if np.any(sel) else RANGE_NONE

    def fnSideRanges(self, msg):
        """Nearest return in each side wedge, as text, for the log (ARX).

        The bitmask alone cannot say whether a call was close or obvious, and
        that is exactly what is wanted when a bay is chosen wrongly.
        """
        n = len(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, n)
        ranges = np.array(msg.ranges)
        ok = (ranges >= msg.range_min) & (ranges <= msg.range_max)
        deg = (np.degrees(angles) + 180.0) % 360.0 - 180.0
        half = self.get_parameter('obstacle.side_angle_deg').value
        out = []
        for centre in (90.0, -90.0):
            sel = ok & (np.abs(deg - centre) <= half)
            out.append(f'{ranges[sel].min():.3f} m' if np.any(sel) else 'nothing')
        return out

    def fnSideName(self, side):
        if side == SIDE_CLEAR:
            return 'both sides clear'
        names = []
        if side & SIDE_LEFT:
            names.append('left')
        if side & SIDE_RIGHT:
            names.append('right')
        return ' and '.join(names) + ' taken'

    def fnBlockedName(self, blocked):
        if blocked == BLOCKED_NONE:
            return 'clear'
        sides = []
        if blocked & BLOCKED_AHEAD:
            sides.append('ahead')
        if blocked & BLOCKED_LEFT:
            sides.append('left')
        if blocked & BLOCKED_RIGHT:
            sides.append('right')
        return ' and '.join(sides)

    def fnPublishImage(self, points, blocked):
        """The example's top-down picture, as a topic rather than a window."""
        img_vis = np.zeros((500, 500, 3), dtype=np.uint8)
        scale = 400.0  # 1 m = 400 pixels
        center = (img_vis.shape[1] // 2, img_vis.shape[0] // 2)

        # The robot, at its real footprint rather than the example's 12 cm
        # square: a Burger is 138 mm across and 178 mm long, and the whole
        # point of the picture is judging clearances by eye.
        robot_width_m = 0.138
        robot_length_m = 0.178
        rw = int(robot_width_m * scale)
        rl = int(robot_length_m * scale)
        tb_top_left = (center[0] - rw // 2, center[1] - rl // 2)
        tb_bottom_right = (center[0] + rw // 2, center[1] + rl // 2)
        cv2.rectangle(img_vis, tb_top_left, tb_bottom_right, (255, 0, 0), 2)

        reach = self.get_parameter('obstacle.stop_distance').value
        half = self.get_parameter('obstacle.half_width').value

        # The three windows fnBlockedHalves actually tests, drawn where they
        # are tested, so a threshold that did not fire can be seen not to
        # have overlapped anything.
        overlay = img_vis.copy()
        for lo, hi, lit in ((-3.0 * half, -half, blocked & BLOCKED_LEFT),
                            (half, 3.0 * half, blocked & BLOCKED_RIGHT),
                            (-half, half, blocked & BLOCKED_AHEAD)):
            top_left = (center[0] + int(lo * scale),
                        center[1] - int(reach * scale))
            bottom_right = (center[0] + int(hi * scale), center[1])
            cv2.rectangle(overlay, top_left, bottom_right,
                          (0, 0, 255) if lit else (0, 160, 0), -1)
        img_vis = cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0)

        # After the blend, so the returns stay at full brightness inside the
        # windows - they are the thing being judged.
        if points is not None and len(points) > 0:
            for pt in points:
                x = int(pt[0] * scale + center[0])
                y = int(-pt[1] * scale + center[1])
                cv2.circle(img_vis, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(img_vis, self.fnBlockedName(blocked), (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if blocked else (0, 255, 0), 2)
        # At 400 px/m the window only reaches 0.625 m, which is easy to
        # forget when nothing appears near the edge.
        cv2.putText(img_vis, f'{500 / 2 / scale:.2f} m to edge', (12, 484),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        self.pub_image.publish(self.bridge.cv2_to_compressed_imgmsg(img_vis, 'jpg'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectObstacle()
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

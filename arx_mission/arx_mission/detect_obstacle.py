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
from std_msgs.msg import UInt8

# ARX: must match mission_control.py.
MISSION_CONSTRUCTION = 3

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


class DetectObstacle(Node):

    def __init__(self):
        super().__init__('detect_obstacle')

        # ARX: how far ahead counts as blocking, and how wide a corridor to
        # look down. See obstacle.yaml for why half_width is narrower than
        # the robot.
        self.declare_parameter('obstacle.stop_distance', 0.25)
        self.declare_parameter('obstacle.half_width', 0.06)
        self.declare_parameter('obstacle.min_points', 3)
        self.declare_parameter('obstacle.frame_skip', 1)
        self.declare_parameter('obstacle.publish_debug_image', True)
        self.declare_parameter('obstacle.always_on', False)

        self.armed = 0
        self.counter = 1
        self.blocked = None

        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        # ARX: the gate every detector in this package has. A scan costs far
        # less than a SIFT frame, but the rules allow props on the course
        # that belong to no mission, and a detector left looking will find
        # one.
        self.create_subscription(UInt8, '/arx/armed_mission', self.cbArmedMission, 1)

        self.pub_obstacle = self.create_publisher(UInt8, '/detect/obstacle', 10)
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
        return self.armed == MISSION_CONSTRUCTION

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

        # ARX: published every scan, not only on a change. mission_control
        # reads the latest value each tick, and a node that comes up late
        # would otherwise wait for the world to change before hearing
        # anything.
        out = UInt8()
        out.data = blocked
        self.pub_obstacle.publish(out)

        if blocked != self.blocked:
            self.blocked = blocked
            self.get_logger().info(f'blocked -> {self.fnBlockedName(blocked)}')

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

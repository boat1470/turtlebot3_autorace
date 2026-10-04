#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
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
# Authors:
#   - Leon Jung, Gilbert, Ashe Kim, ChanHyeong Lee
#   - [AuTURBO] Kihoon Kim (https://github.com/auturbo)
#
# ARX: the idea is turtlebot3_autorace_detect/detect_level_crossing.py's, and
# it is a good one: the bar is painted in red and white bands, so it shows up
# as a row of red blobs, and the ANGLE of the line through them says whether
# the bar is lying across the road or standing up out of the way. An angle is
# the right thing to measure because it does not depend on where the robot is
# standing - no calibration of position, no assumption about which part of the
# frame the bar will appear in.
#
# What is not kept, and why. Every number below was measured on this course
# with this camera, at y 1.25 facing +x, against the bar at (-0.85, 1.26).
#
#   1. minArea 150 / maxArea 500. Run against real frames at six distances,
#      that pair sees the bar over a window 4 cm wide:
#
#        distance   0.51  0.55  0.60  0.65  0.70  0.75 m
#        blobs         3     3     2     0     0     0
#
#      Below 150 px^2 every band is dropped, so beyond 0.60 m the node
#      reports nothing - and reporting nothing takes the `len <= 1` branch,
#      which says the crossing is OPEN. A bar that is down and 0.65 m away
#      reads as clear to go.
#
#   2. `len(keypts) == 3`. The bar's own texture has FOUR red bands, and 4 or
#      5 blobs is what these frames actually give. The original has branches
#      for 3 and for <= 1 and nothing in between, so 2 or 4 blobs fall
#      through every branch, leave all three flags False, and the caller
#      waits on a result that never comes.
#
#   3. `is_rects_linear or is_rects_dist_equal`. Straightness is judged at 50
#      px and even spacing at 6; with `or`, the loose one passes first and the
#      tight one is never asked. Lowering minArea makes this worse, not
#      better - at 0.35 m a low threshold brings in 7 blobs where the bar has
#      4, and their spacing has a standard deviation of 17.6 px. Both tests
#      have to hold, and the extras have to be dropped before either is
#      applied.
#
#   4. The whole of level_crossing_order: three `while rclpy.ok()` loops
#      calling rclpy.spin_once from inside a subscriber callback, publishing
#      to a `pub_max_vel` that is never created. It has never run - nothing in
#      the workspace publishes /detect/level_crossing_order, because the node
#      that used to is ROS 1 and was not ported. This node reports and decides
#      nothing, like every other detector here.
#
#   5. Hue 0-22 alone. Red straddles the 0/180 wrap in OpenCV's HSV, so one
#      range catches half of it: a band that renders at hue 167 is dropped
#      while orange at hue 15 is let in. Two ranges, OR'd.
#
# What LiDAR would give: nothing. The bar's collision box spans z 0.100 to
# 0.150 and the LDS sits at z 0.171, so the beam passes 21 mm over the top of
# it. This has to be done with the camera.

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import UInt8

# ARX: must match mission_control.py.
MISSION_LEVEL = 5

# ARX: what this node publishes on /detect/level_bar.
BAR_NONE = 0
BAR_UP = 1
BAR_DOWN = 2
BAR_NAMES = {BAR_NONE: 'nothing', BAR_UP: 'up', BAR_DOWN: 'down'}

# ARX: what it publishes on /detect/level_range when it has nothing to
# measure. Large rather than zero, so a mission comparing `range <= stop_m`
# does not read silence as "arrived".
RANGE_NONE = 9.9


class DetectLevel(Node):

    def __init__(self):
        super().__init__('detect_level')

        # ARX: two red ranges. See the note at the top - one range catches
        # half of red, because red straddles the 0/180 wrap.
        self.declare_parameter('level.red.hue_lo_l', 0)
        self.declare_parameter('level.red.hue_lo_h', 10)
        self.declare_parameter('level.red.hue_hi_l', 160)
        self.declare_parameter('level.red.hue_hi_h', 179)
        self.declare_parameter('level.red.saturation_l', 173)
        self.declare_parameter('level.red.saturation_h', 255)
        self.declare_parameter('level.red.lightness_l', 106)
        self.declare_parameter('level.red.lightness_h', 255)

        # ARX: the blob detector. minArea is the one number that decides how
        # far away the bar can still be seen - see the table at the top.
        self.declare_parameter('level.min_area', 80.0)
        self.declare_parameter('level.max_area', 5000.0)
        self.declare_parameter('level.min_convexity', 0.7)

        # ARX: which blobs belong to the bar. Its bands sit on one line, so
        # they share an image row to within a few pixels while a stray red
        # thing elsewhere in the frame does not. Grouped by row first, and the
        # largest group is the bar; everything else is dropped before any
        # geometry is attempted. Without this the line fit is pulled off by
        # blobs that were never part of the bar.
        self.declare_parameter('level.row_band_px', 25.0)
        self.declare_parameter('level.min_blobs', 3)

        # ARX: the two checks the original had, with the thresholds brought
        # onto the same scale and joined by `and` rather than `or`.
        self.declare_parameter('level.max_line_error_px', 6.0)
        self.declare_parameter('level.max_gap_spread_px', 6.0)

        # ARX: the original's rule, kept as it is. |slope| of the fitted line
        # in image coordinates - flat means the bar is lying across the road,
        # steep means it is standing up. 2.0 is 63.4 degrees from horizontal,
        # which splits the two unevenly on purpose: 63 degrees of tolerance
        # for "still down" against 27 for "up", because calling a lowered bar
        # raised is the expensive mistake. Measured here: 0.03 with the bar
        # down and 15.72 with it up, so there is a factor of 500 between the
        # two answers and the threshold has room either side.
        self.declare_parameter('level.up_slope', 2.0)

        # ARX: pixels times metres. The bar's apparent width is inversely
        # proportional to distance, so span_px * distance is a constant of the
        # camera and the bar. Half the bar measured 13.49 and 12.44 px.m at
        # two distances, so the full span is about 26 - but two points is not
        # a calibration, and this wants re-measuring from a run before it is
        # trusted. Until then, /detect/level_bar is the part worth acting on.
        self.declare_parameter('level.k_px_m', 29.0)

        self.declare_parameter('level.frame_skip', 1)
        self.declare_parameter('level.publish_debug_image', True)
        self.declare_parameter('level.always_on', False)

        self.armed = 0
        self.counter = 1
        self.bar = None
        self.cv_bridge = CvBridge()

        self.create_subscription(Image, '/detect/image_input', self.on_image, 1)
        # ARX: the gate every detector in this package has. The course carries
        # red on other props, and a detector left looking will find some.
        self.create_subscription(UInt8, '/arx/armed_mission', self.on_armed, 1)

        self.pub_bar = self.create_publisher(UInt8, '/detect/level_bar', 10)
        self.pub_range = self.create_publisher(Float32, '/detect/level_range', 10)
        self.pub_image = self.create_publisher(
            CompressedImage, '/detect/image_output/compressed', 10)

        self.get_logger().info('DetectLevel Node Initialized')

    def on_armed(self, msg):
        """Remember which mission the sequencer says is running (ARX)."""
        if msg.data == self.armed:
            return
        self.armed = msg.data
        self.get_logger().info(f'armed mission -> {self.armed}')

    def fnArmed(self):
        """Whether this detector should be looking at all (ARX)."""
        if self.get_parameter('level.always_on').value:
            return True
        return self.armed == MISSION_LEVEL

    def on_image(self, msg):
        if not self.fnArmed():
            return

        frame_skip = self.get_parameter('level.frame_skip').value
        if frame_skip > 1:
            if self.counter % frame_skip != 0:
                self.counter += 1
                return
            self.counter = 1

        image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        mask = self.fnMaskRed(image)
        blobs = self.fnFindBlobs(mask)
        bar, span, chosen, line = self.fnReadBar(blobs)

        out = UInt8()
        out.data = bar
        self.pub_bar.publish(out)

        rng = Float32()
        if bar == BAR_DOWN and span > 0.0:
            rng.data = float(self.get_parameter('level.k_px_m').value / span)
        else:
            rng.data = RANGE_NONE
        self.pub_range.publish(rng)

        if bar != self.bar:
            self.bar = bar
            where = ('' if bar != BAR_DOWN
                     else f', {rng.data:.2f} m away, span {span:.1f} px')
            self.get_logger().info(
                f'level bar -> {BAR_NAMES[bar]} '
                f'({len(blobs)} blobs, {len(chosen)} on the bar{where})')

        if self.get_parameter('level.publish_debug_image').value:
            self.fnPublishImage(image, blobs, chosen, line, bar, rng.data)

    def fnMaskRed(self, image):
        """Both halves of red, inverted for the blob detector (ARX)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_l = self.get_parameter('level.red.saturation_l').value
        s_h = self.get_parameter('level.red.saturation_h').value
        v_l = self.get_parameter('level.red.lightness_l').value
        v_h = self.get_parameter('level.red.lightness_h').value
        lo = cv2.inRange(
            hsv,
            np.array([self.get_parameter('level.red.hue_lo_l').value, s_l, v_l]),
            np.array([self.get_parameter('level.red.hue_lo_h').value, s_h, v_h]))
        hi = cv2.inRange(
            hsv,
            np.array([self.get_parameter('level.red.hue_hi_l').value, s_l, v_l]),
            np.array([self.get_parameter('level.red.hue_hi_h').value, s_h, v_h]))
        self.mask_red = cv2.bitwise_or(lo, hi)
        # Inverted because SimpleBlobDetector defaults to filterByColor with
        # blobColor 0 - it looks for DARK blobs - so the bands have to be dark
        # on a light field. Blurred because the detector works by thresholding
        # at many levels and keeping what survives all of them, and a mask of
        # nothing but 0 and 255 gives the same picture at every level, so that
        # step has nothing to work with.
        return cv2.GaussianBlur(cv2.bitwise_not(self.mask_red), (5, 5), 0)

    def fnFindBlobs(self, mask):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = self.get_parameter('level.min_area').value
        params.maxArea = self.get_parameter('level.max_area').value
        params.filterByConvexity = True
        params.minConvexity = self.get_parameter('level.min_convexity').value
        # ARX: off. It defaults to ON with minInertiaRatio 0.1, which drops
        # any blob more than ten times longer than it is wide - and a band of
        # the bar seen at an angle is exactly that. The original never set it
        # and so never knew it was there.
        params.filterByInertia = False
        return cv2.SimpleBlobDetector_create(params).detect(mask)

    def fnReadBar(self, blobs):
        """The bar's state, and how wide it is in pixels (ARX).

        Returns (state, span_px, blobs on the bar, fitted line or None).
        """
        if len(blobs) < self.get_parameter('level.min_blobs').value:
            return BAR_NONE, 0.0, [], None

        chosen = self.fnLargestRow(blobs)
        if len(chosen) < self.get_parameter('level.min_blobs').value:
            return BAR_NONE, 0.0, chosen, None

        pts = np.array([b.pt for b in chosen], dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()

        # Distance of each point from the fitted line - the same question
        # fnCheckLinearity asked of the middle point, asked of all of them.
        off = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        # Where each point falls ALONG the line, so the bands can be sorted
        # and their spacing compared whatever angle the bar is at.
        along = np.sort((pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy)
        gaps = np.diff(along)

        line = (float(vx), float(vy), float(x0), float(y0))
        if off.max() > self.get_parameter('level.max_line_error_px').value:
            return BAR_NONE, 0.0, chosen, line
        if gaps.size > 1 and gaps.std() > self.get_parameter(
                'level.max_gap_spread_px').value:
            return BAR_NONE, 0.0, chosen, line

        slope = abs(vy / vx) if abs(vx) > 1e-6 else float('inf')
        if slope > self.get_parameter('level.up_slope').value:
            return BAR_UP, 0.0, chosen, line
        return BAR_DOWN, float(along[-1] - along[0]), chosen, line

    def fnLargestRow(self, blobs):
        """The biggest group of blobs sharing an image row (ARX).

        The bar's bands lie on one line, so they land within a few pixels of
        each other in y however far away it is; a red thing somewhere else in
        the frame does not. Grouping first is what lets min_area come down far
        enough to see the bar at 0.75 m without the strays that come with it
        dragging the line fit off.
        """
        band = self.get_parameter('level.row_band_px').value
        best = []
        for anchor in blobs:
            group = [b for b in blobs if abs(b.pt[1] - anchor.pt[1]) <= band]
            if len(group) > len(best):
                best = group
        return best

    def fnPublishImage(self, image, blobs, chosen, line, bar, rng):
        frame = cv2.drawKeypoints(
            image, blobs, np.array([]), (90, 90, 90),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # The ones that survived the row grouping, over the top, so a wrong
        # call can be read off the picture: grey was seen and dropped, yellow
        # was used.
        frame = cv2.drawKeypoints(
            frame, chosen, np.array([]), (0, 255, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if line is not None and len(chosen) >= 2:
            vx, vy, x0, y0 = line
            t = [(p.pt[0] - x0) * vx + (p.pt[1] - y0) * vy for p in chosen]
            a = (int(x0 + vx * min(t)), int(y0 + vy * min(t)))
            b = (int(x0 + vx * max(t)), int(y0 + vy * max(t)))
            cv2.line(frame, a, b, (255, 0, 0), 2)
        text = BAR_NAMES[bar] + ('' if bar != BAR_DOWN else f'  {rng:.2f} m')
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if bar == BAR_DOWN else (0, 255, 0), 2)
        self.pub_image.publish(
            self.cv_bridge.cv2_to_compressed_imgmsg(frame, 'jpg'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectLevel()
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

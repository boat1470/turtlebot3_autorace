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
# Authors:
#   - Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
#   - Special Thanks : Roger Sacchelli
#
# Copyright 2026 boat1470 (changes marked # ARX)
#
# ARX: forked from turtlebot3_autorace_detect/detect_lane.py. Same file name,
# same class, same method names, same variable names, so a diff against the
# original shows only what changed. Every change is marked # ARX. Six of them:
#
#   1. make_lane could reach the publish with centerx never assigned, which
#      raised NameError inside the subscription callback and took the node
#      down. See the note there - this is the one that could end a run.
#   2. The fits are no longer conjured by a bare `except Exception` around
#      both lanes at once.
#   3. frame_skip is a parameter instead of a literal % 3.
#   4. The mask debug images no longer require calibration mode, which also
#      switched the lightness auto-adjust over.
#   5. The parameter callback is registered in every mode, and it no longer
#      returns after the first parameter in the list.
#   6. The 600-iteration Python loop counting non-empty mask rows is a numpy
#      call, and does not assume the image is 600 rows tall.

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8

# ARX: /arx/follow_side - must match mission_control.py.
FOLLOW_AUTO = 0
FOLLOW_YELLOW = 1
FOLLOW_WHITE = 2
SIDE_NAMES = {FOLLOW_AUTO: 'auto', FOLLOW_YELLOW: 'yellow', FOLLOW_WHITE: 'white'}


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        parameter_descriptor_hue = ParameterDescriptor(
            description='hue parameter range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=179,
                step=1)]
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            description='saturation and lightness range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=255,
                step=1)]
        )
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detect.lane.white.hue_l', 0,
                    parameter_descriptor_hue),
                ('detect.lane.white.hue_h', 179,
                    parameter_descriptor_hue),
                ('detect.lane.white.saturation_l', 0,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.saturation_h', 70,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_l', 105,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.hue_l', 10,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.hue_h', 127,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.saturation_l', 70,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.saturation_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_l', 95,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('is_detection_calibration_mode', False),
                # ARX: the original had no parameters beyond the HSV bands
                # and the calibration flag. These four were literals or were
                # bolted onto is_detection_calibration_mode.
                ('lane.frame_skip', 3),
                ('lane.publish_debug_image', False),
                ('lane.auto_adjust_white', True),
                ('lane.auto_adjust_yellow', False),
                ('lane.log_lane_state', False),
            ]
        )

        self.hue_white_l = self.get_parameter(
            'detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = self.get_parameter(
            'detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = self.get_parameter(
            'detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = self.get_parameter(
            'detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.lightness_white_l = self.get_parameter(
            'detect.lane.white.lightness_l').get_parameter_value().integer_value
        self.lightness_white_h = self.get_parameter(
            'detect.lane.white.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'detect.lane.yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'detect.lane.yellow.lightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_detection_calibration_mode').get_parameter_value().bool_value
        # ARX: registered in every mode. The original tied it to calibration
        # mode, so retuning a threshold live meant also turning on the
        # lightness auto-adjust and the mask images - three unrelated things
        # behind one flag.
        self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        # ARX: was self.is_calibration_mode.
        self.publish_debug_image = self.get_parameter(
            'lane.publish_debug_image').get_parameter_value().bool_value

        self.sub_image_type = 'raw'         # you can choose image type 'compressed', 'raw'
        self.pub_image_type = 'compressed'  # you can choose image type 'compressed', 'raw'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1
                )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1
                )

        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1
                )
        elif self.pub_image_type == 'raw':
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1
                )

        if self.publish_debug_image:  # ARX: was self.is_calibration_mode
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1
                    )
            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1
                    )

        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)

        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8, '/detect/yellow_line_reliability', 1
            )

        self.pub_white_line_reliability = self.create_publisher(
            UInt8, '/detect/white_line_reliability', 1
            )

        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        # ARX: which line to steer by, once the junction has been decided.
        #
        # Normally the centre is the mean of both lines, which keeps the
        # robot in the middle of whatever it is already in. That is the wrong
        # thing at a fork: both branches are lane, so the mean simply splits
        # the difference and the robot goes wherever the geometry happens to
        # push it.
        #
        # Steering by one line instead picks a branch, because the line
        # itself goes into one of them. The arithmetic is the same the node
        # already uses when only one line is visible - the line, plus or
        # minus half a lane width - so following the yellow line is exactly
        # what this node does anyway when the white one drops out.
        self.follow_side = FOLLOW_AUTO
        self.create_subscription(UInt8, '/arx/follow_side', self.cbFollowSide, 1)

        self.cvBridge = CvBridge()

        self.counter = 1

        self.window_width = 1000.
        self.window_height = 600.

        self.reliability_white_line = 100
        self.reliability_yellow_line = 100

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))

        # ARX: the original never created these. It read self.left_fit on the
        # first frame, caught the AttributeError, and used that to mean "no
        # previous fit yet" - inside a bare `except Exception` wrapped around
        # both lanes, which then swallowed every genuine failure in
        # fit_from_lines for the rest of the run, silently.
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        # ARX: was a single attribute shared by both lanes, so a failed
        # polyfit on one lane fell back to the other lane's curve.
        self.lane_fit_bef = {'left': None, 'right': None}

    def cbFollowSide(self, msg):
        """Latch which line to steer by (ARX)."""
        if msg.data == self.follow_side:
            return
        self.follow_side = msg.data
        self.get_logger().info(
            'steering by the mean of both lines' if msg.data == FOLLOW_AUTO
            else f'steering by the {SIDE_NAMES.get(msg.data, msg.data)} line')

    def cbGetDetectLaneParam(self, parameters):
        for param in parameters:
            self.get_logger().info(f'Parameter name: {param.name}')
            self.get_logger().info(f'Parameter value: {param.value}')
            self.get_logger().info(f'Parameter type: {param.type_}')
            if param.name == 'detect.lane.white.hue_l':
                self.hue_white_l = param.value
            elif param.name == 'detect.lane.white.hue_h':
                self.hue_white_h = param.value
            elif param.name == 'detect.lane.white.saturation_l':
                self.saturation_white_l = param.value
            elif param.name == 'detect.lane.white.saturation_h':
                self.saturation_white_h = param.value
            elif param.name == 'detect.lane.white.lightness_l':
                self.lightness_white_l = param.value
            elif param.name == 'detect.lane.white.lightness_h':
                self.lightness_white_h = param.value
            elif param.name == 'detect.lane.yellow.hue_l':
                self.hue_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.hue_h':
                self.hue_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.saturation_l':
                self.saturation_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.saturation_h':
                self.saturation_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.lightness_l':
                self.lightness_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.lightness_h':
                self.lightness_yellow_h = param.value
        # ARX: this return was indented into the for loop, so setting several
        # parameters in one call applied the first and dropped the rest
        # without saying so. rqt_reconfigure sends them one at a time, which
        # is why it never showed.
        return SetParametersResult(successful=True)

    def cbFindLane(self, image_msg):
        # Change the frame rate by yourself. Now, it is set to 1/3 (10fps).
        # Unappropriate value of frame rate may cause huge delay on entire recognition process.
        # This is up to your computer's operating power.
        # ARX: a parameter now, and 0 means look at every frame.
        frame_skip = self.get_parameter('lane.frame_skip').value
        if frame_skip > 1:
            if self.counter % frame_skip != 0:
                self.counter += 1
                return
            else:
                self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        white_fraction, cv_white_lane = self.maskWhiteLane(cv_image)
        yellow_fraction, cv_yellow_lane = self.maskYellowLane(cv_image)

        # ARX: one try/except covered both lanes, so a failure while fitting
        # the right lane also re-fitted the left one from scratch, throwing
        # away a good curve it had just computed. Each lane now stands alone,
        # and the "have I got a previous fit to search near" question is
        # asked directly instead of being inferred from an AttributeError.
        if yellow_fraction > 3000:
            self.fnUpdateLane('left', cv_yellow_lane)

        if white_fraction > 3000:
            self.fnUpdateLane('right', cv_white_lane)

        # ARX: with no fit at all there is nothing to average or to draw.
        if self.left_fit is None and self.right_fit is None:
            return

        MOV_AVG_LENGTH = 5

        # ARX: guarded. The original indexed these unconditionally, which is
        # an IndexError on an empty array - reachable whenever one lane has
        # never been seen, because the shared try/except above always left
        # both arrays in step with each other and these never checked.
        if self.mov_avg_left.shape[0] > 0:
            self.left_fit = np.array([
                np.mean(self.mov_avg_left[-MOV_AVG_LENGTH:][:, 0]),
                np.mean(self.mov_avg_left[-MOV_AVG_LENGTH:][:, 1]),
                np.mean(self.mov_avg_left[-MOV_AVG_LENGTH:][:, 2])
                ])
        if self.mov_avg_right.shape[0] > 0:
            self.right_fit = np.array([
                np.mean(self.mov_avg_right[-MOV_AVG_LENGTH:][:, 0]),
                np.mean(self.mov_avg_right[-MOV_AVG_LENGTH:][:, 1]),
                np.mean(self.mov_avg_right[-MOV_AVG_LENGTH:][:, 2])
                ])

        # Keep the NEWEST rows, not the oldest. The moving average above reads
        # mov_avg[::-1][0:MOV_AVG_LENGTH], i.e. the most recent frames, so
        # trimming to [0:MOV_AVG_LENGTH] hands it five frames from a thousand
        # frames ago - about two and a half minutes at the rate this runs.
        if self.mov_avg_left.shape[0] > 1000:
            self.mov_avg_left = self.mov_avg_left[-MOV_AVG_LENGTH:]

        if self.mov_avg_right.shape[0] > 1000:
            self.mov_avg_right = self.mov_avg_right[-MOV_AVG_LENGTH:]

        self.make_lane(cv_image, white_fraction, yellow_fraction)

    def maskWhiteLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_white_l
        Hue_h = self.hue_white_h
        Saturation_l = self.saturation_white_l
        Saturation_h = self.saturation_white_h
        Lightness_l = self.lightness_white_l
        Lightness_h = self.lightness_white_h

        lower_white = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_white = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        fraction_num = np.count_nonzero(mask)

        # ARX: was `if not self.is_calibration_mode`. White adjusted while
        # racing and yellow adjusted while calibrating - the two are written
        # the opposite way round in the original, which is almost certainly a
        # typo. Both are parameters now, defaulted to the original behaviour
        # so this fork drives the same as what was tuned.
        if self.get_parameter('lane.auto_adjust_white').value:
            if fraction_num > 35000:
                if self.lightness_white_l < 250:
                    self.lightness_white_l += 5
            elif fraction_num < 5000:
                if self.lightness_white_l > 50:
                    self.lightness_white_l -= 5

        # ARX: was a 600-iteration Python loop calling count_nonzero on
        # each row, twice per frame. Same result, and it no longer assumes
        # the projected image is exactly 600 rows tall.
        how_much_short = mask.shape[0] - np.count_nonzero(mask.any(axis=1))

        if how_much_short > 100:
            if self.reliability_white_line >= 5:
                self.reliability_white_line -= 5
        elif how_much_short <= 100:
            if self.reliability_white_line <= 99:
                self.reliability_white_line += 5

        msg_white_line_reliability = UInt8()
        msg_white_line_reliability.data = self.reliability_white_line
        self.pub_white_line_reliability.publish(msg_white_line_reliability)

        if self.publish_debug_image:  # ARX: was self.is_calibration_mode
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
                    )

            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
                    )

        return fraction_num, mask

    def maskYellowLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_yellow_l
        Hue_h = self.hue_yellow_h
        Saturation_l = self.saturation_yellow_l
        Saturation_h = self.saturation_yellow_h
        Lightness_l = self.lightness_yellow_l
        Lightness_h = self.lightness_yellow_h

        lower_yellow = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_yellow = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        fraction_num = np.count_nonzero(mask)

        # ARX: see maskWhiteLane. Was `if self.is_calibration_mode`.
        if self.get_parameter('lane.auto_adjust_yellow').value:
            if fraction_num > 35000:
                if self.lightness_yellow_l < 250:
                    self.lightness_yellow_l += 20
            elif fraction_num < 5000:
                if self.lightness_yellow_l > 90:
                    self.lightness_yellow_l -= 20

        # ARX: was a 600-iteration Python loop calling count_nonzero on
        # each row, twice per frame. Same result, and it no longer assumes
        # the projected image is exactly 600 rows tall.
        how_much_short = mask.shape[0] - np.count_nonzero(mask.any(axis=1))

        if how_much_short > 100:
            if self.reliability_yellow_line >= 5:
                self.reliability_yellow_line -= 5
        elif how_much_short <= 100:
            if self.reliability_yellow_line <= 99:
                self.reliability_yellow_line += 5

        msg_yellow_line_reliability = UInt8()
        msg_yellow_line_reliability.data = self.reliability_yellow_line
        self.pub_yellow_line_reliability.publish(msg_yellow_line_reliability)

        if self.publish_debug_image:  # ARX: was self.is_calibration_mode
            if self.pub_image_type == 'compressed':
                self.pub_image_yellow_lane.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
                    )

            elif self.pub_image_type == 'raw':
                self.pub_image_yellow_lane.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
                    )

        return fraction_num, mask

    def fnUpdateLane(self, left_or_right, image):
        """Refit one lane and push the result into its moving average (ARX).

        Replaces the try/except that wrapped both lanes in the original. The
        order is the same one it produced: search near the previous fit when
        there is one, fall back to the sliding window when there is not or
        when that search comes up empty.

        Returns nothing. A frame that yields no fit leaves the previous one
        in place, and make_lane decides whether it is still usable.
        """
        lane_fit = self.left_fit if left_or_right == 'left' else self.right_fit

        lane_fitx = None
        extend = False
        if lane_fit is not None:
            lane_fitx, lane_fit = self.fit_from_lines(lane_fit, image)
            extend = lane_fitx is not None

        if lane_fitx is None:
            lane_fitx, lane_fit = self.sliding_windown(image, left_or_right)

        if lane_fitx is None:
            return

        row = np.array([lane_fit])
        if left_or_right == 'left':
            self.left_fitx, self.left_fit = lane_fitx, lane_fit
            self.mov_avg_left = (
                np.append(self.mov_avg_left, row, axis=0) if extend else row)
        else:
            self.right_fitx, self.right_fit = lane_fitx, lane_fit
            self.mov_avg_right = (
                np.append(self.mov_avg_right, row, axis=0) if extend else row)

    def fit_from_lines(self, lane_fit, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_inds = (
            (nonzerox >
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
            (nonzerox <
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin))
                )

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # ARX: polyfit raises on an empty or near-empty vector, and the
        # original let that escape into the caller's bare except. Saying "no
        # fit" out loud is the same answer without hiding anything else.
        if len(y) < 3:
            return None, None

        lane_fit = np.polyfit(y, x, 2)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def sliding_windown(self, img_w, left_or_right):
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)

        out_img = np.dstack((img_w, img_w, img_w)) * 255

        midpoint = np.int_(histogram.shape[0] / 2)

        if left_or_right == 'left':
            lane_base = np.argmax(histogram[:midpoint])
        elif left_or_right == 'right':
            lane_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 20

        window_height = np.int_(img_w.shape[0] / nwindows) # 600/20 = 30

        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x_current = lane_base

        margin = 50

        minpix = 50

        lane_inds = []

        for window in range(nwindows):
            win_y_low = img_w.shape[0] - (window + 1) * window_height   #600 - (1-20)*30
            win_y_high = img_w.shape[0] - window * window_height        #600 - (0-19)*30
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            cv2.rectangle(
                out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            good_lane_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) &
                (nonzerox < win_x_high)
                ).nonzero()[0]

            lane_inds.append(good_lane_inds)

            if len(good_lane_inds) > minpix:
                x_current = np.int_(np.mean(nonzerox[good_lane_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            # ARX: keyed by lane. A single shared attribute meant a failed
            # fit on one lane fell back to the other lane's curve, which is
            # 280 px away by construction.
            self.lane_fit_bef[left_or_right] = lane_fit
        except Exception:
            lane_fit = self.lane_fit_bef[left_or_right]
            # ARX: and on the very first frame there is nothing to fall back
            # to. The original raised AttributeError here.
            if lane_fit is None:
                return None, None

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def make_lane(self, cv_image, white_fraction, yellow_fraction):
        # Create an image to draw the lines on
        warp_zero = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), dtype=np.uint8)

        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])

        # both lane -> 2, left lane -> 1, right lane -> 3, none -> 0
        lane_state = UInt8()

        # ARX: the original tested the pixel counts and then indexed
        # self.left_fitx / self.right_fitx as though a count above 3000
        # guaranteed a curve. It does not: a frame can be full of yellow and
        # still produce no fit polyfit will accept. Ask both questions.
        has_left = yellow_fraction > 3000 and self.left_fitx is not None
        has_right = white_fraction > 3000 and self.right_fitx is not None

        if has_left:  # ARX
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_left]),
                isClosed=False,
                color=(0, 0, 255),
                thickness=25
                )

        if has_right:  # ARX
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_right]),
                isClosed=False,
                color=(255, 255, 0),
                thickness=25
                )

        self.is_center_x_exist = True
        # ARX: this is the crash. The original set is_center_x_exist True
        # here and then relied on one of the branches below assigning centerx
        # - but inside the first branch all three tests can be false at once
        # (both fractions at or below 3000 while neither reliability has
        # decayed past 50), and then nothing assigns centerx and nothing
        # clears the flag. The publish at the bottom reads it anyway.
        #
        # UnboundLocalError, raised inside a subscription callback, which
        # rclpy lets propagate out of spin. The node exits; /detect/lane goes
        # quiet; control_lane stops publishing because it only publishes when
        # a lane message arrives; the diff drive plugin holds the last
        # velocity it was given. The robot drives on in a straight line with
        # nothing left running to notice.
        #
        # Reliability falls by 5 a frame from 100, so the window where both
        # lines are gone and both reliabilities are still above 50 is ten
        # frames wide - around three seconds at this frame rate.
        #
        # Reproduced against the unmodified example with eight blank frames
        # after a run of good ones.
        centerx = None

        # ARX: a decided junction overrides everything below. Half a lane
        # width from the chosen line, which is the same arithmetic the
        # branches below use when only that line is visible.
        #
        # The condition includes has_left / has_right, so a frame where the
        # chosen line is missing falls through to the normal logic rather
        # than reporting nothing. Losing sight of it for a few frames on a
        # curve is ordinary; refusing to steer at all until it comes back
        # would not be.
        if self.follow_side == FOLLOW_YELLOW and has_left:
            centerx = np.add(self.left_fitx, 280)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            lane_state.data = 1
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        elif self.follow_side == FOLLOW_WHITE and has_right:
            centerx = np.subtract(self.right_fitx, 280)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            lane_state.data = 3
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        elif self.reliability_white_line > 50 and self.reliability_yellow_line > 50:
            if has_right and has_left:  # ARX
                centerx = np.mean([self.left_fitx, self.right_fitx], axis=0)
                pts = np.hstack((pts_left, pts_right))
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 2

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

                # Draw the lane onto the warped blank image
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            if has_right and not has_left:  # ARX
                centerx = np.subtract(self.right_fitx, 280)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 3

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

            if has_left and not has_right:  # ARX
                centerx = np.add(self.left_fitx, 280)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 1

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

        elif (self.reliability_white_line <= 50 and
                self.reliability_yellow_line > 50 and
                has_left):  # ARX: and a left curve to add 280 to
            centerx = np.add(self.left_fitx, 280)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 1

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        elif (self.reliability_white_line > 50 and
                self.reliability_yellow_line <= 50 and
                has_right):  # ARX: and a right curve to subtract 280 from
            centerx = np.subtract(self.right_fitx, 280)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 3

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        else:
            self.is_center_x_exist = False

            lane_state.data = 0

            pass

        # ARX: the guard. Whatever route was taken above, if it did not leave
        # a centre line then this frame has no lane to report - say so rather
        # than reading a name that was never bound.
        if centerx is None:
            self.is_center_x_exist = False
            lane_state.data = 0

        self.pub_lane_state.publish(lane_state)
        # ARX: off by default. This ran on every processed frame at info
        # severity, which is rosout traffic on the Pi for a number
        # /detect/lane_state already carries.
        if self.get_parameter('lane.log_lane_state').value:
            self.get_logger().info(f'Lane state: {lane_state.data}')

        # Combine the result with the original image
        final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
        final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)

        if self.pub_image_type == 'compressed':
            if self.is_center_x_exist:
                # publishes lane center
                msg_desired_center = Float64()
                msg_desired_center.data = centerx.item(350)
                self.pub_lane.publish(msg_desired_center)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))

        elif self.pub_image_type == 'raw':
            if self.is_center_x_exist:
                # publishes lane center
                msg_desired_center = Float64()
                msg_desired_center.data = centerx.item(350)
                self.pub_lane.publish(msg_desired_center)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(final, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

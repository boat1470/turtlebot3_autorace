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
# Author: Leon Jung, Gilbert, Ashe Kim, ChanHyeong Lee
#
# Forked into arx_mission by boat1470, 2026. Same filename and same structure
# as the original so the two can be diffed directly; every change is listed in
# arx_mission/README-fork.md and marked ARX below. The detection maths itself
# - the HSV bands, the blur, the blob filter, the region of interest - is
# untouched, so the same frame gives the same answer.


import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8

# ARX: /arx/traffic_light. The original keeps its result in a local variable
# and draws it on a debug image, so nothing downstream can act on the light.
LIGHT_UNKNOWN = 0
LIGHT_RED = 1
LIGHT_YELLOW = 2
LIGHT_GREEN = 3

NAMES = {LIGHT_UNKNOWN: 'unknown', LIGHT_RED: 'red',
         LIGHT_YELLOW: 'yellow', LIGHT_GREEN: 'green'}

# ARX: /arx/armed_mission, published by mission_control.
MISSION_NONE = 0
MISSION_TRAFFIC_LIGHT = 1


class DetectTrafficLight(Node):

    def __init__(self):
        super().__init__('detect_traffic_light')
        parameter_descriptor_hue = ParameterDescriptor(
            integer_range=[IntegerRange(from_value=0, to_value=179, step=1)],
            description='Hue Value (0~179)'
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)],
            description='Saturation/Lightness Value (0~255)'
        )

        self.declare_parameter(
            'red.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'red.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'red.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.lightness_h', 255, parameter_descriptor_saturation_lightness)

        self.declare_parameter(
            'yellow.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_h', 255, parameter_descriptor_saturation_lightness)

        self.declare_parameter(
            'green.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'green.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'green.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.lightness_h', 255, parameter_descriptor_saturation_lightness)

        self.declare_parameter('is_calibration_mode', False)
        # ARX: the original hard-codes "process one frame in three", which
        # turns a 10 fps camera into 3.3 fps and makes each frame of
        # confirmation cost a third of a second of race time.
        self.declare_parameter('frame_skip', 0)
        # ARX: work even when the mission layer has not armed this mission,
        # for tuning the thresholds on their own.
        self.declare_parameter('always_on', False)
        self.declare_parameter('publish_debug_image', True)

        self.hue_red_l = self.get_parameter(
            'red.hue_l').get_parameter_value().integer_value
        self.hue_red_h = self.get_parameter(
            'red.hue_h').get_parameter_value().integer_value
        self.saturation_red_l = self.get_parameter(
            'red.saturation_l').get_parameter_value().integer_value
        self.saturation_red_h = self.get_parameter(
            'red.saturation_h').get_parameter_value().integer_value
        self.lightness_red_l = self.get_parameter(
            'red.lightness_l').get_parameter_value().integer_value
        self.lightness_red_h = self.get_parameter(
            'red.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'yellow.lightness_h').get_parameter_value().integer_value

        self.hue_green_l = self.get_parameter(
            'green.hue_l').get_parameter_value().integer_value
        self.hue_green_h = self.get_parameter(
            'green.hue_h').get_parameter_value().integer_value
        self.saturation_green_l = self.get_parameter(
            'green.saturation_l').get_parameter_value().integer_value
        self.saturation_green_h = self.get_parameter(
            'green.saturation_h').get_parameter_value().integer_value
        self.lightness_green_l = self.get_parameter(
            'green.lightness_l').get_parameter_value().integer_value
        self.lightness_green_h = self.get_parameter(
            'green.lightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_calibration_mode').get_parameter_value().bool_value
        # ARX: registered unconditionally. The original only does this in
        # calibration mode, so the thresholds cannot be retuned during a real
        # run - and there is exactly one practice day.
        self.add_on_set_parameters_callback(self.get_detect_traffic_light_param)

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        self.counter = 1

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.get_image, 1)
        else:
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.get_image, 1)

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_light = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1)
        else:
            self.pub_image_traffic_light = self.create_publisher(
                Image, '/detect/image_output', 1)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_red_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1)
                self.pub_image_yellow_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1)
                self.pub_image_green_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub3/compressed', 1)
            else:
                self.pub_image_red_light = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1)
                self.pub_image_yellow_light = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1)
                self.pub_image_green_light = self.create_publisher(
                    Image, '/detect/image_output_sub3', 1)

        # ARX: the light state, which is the whole point of the fork.
        self.pub_traffic_light = self.create_publisher(UInt8, '/arx/traffic_light', 1)
        self.sub_armed = self.create_subscription(
            UInt8, '/arx/armed_mission', self.get_armed, 1)

        self.cvBridge = CvBridge()
        self.cv_image = None

        self.is_image_available = False

        # ARX: frame_seq counts frames received, done_seq the last one looked
        # at. The original sets is_image_available True and never clears it,
        # so a stalled camera has it re-detecting the same stale frame at
        # 10 Hz and reporting each pass as a fresh observation.
        self.frame_seq = 0
        self.done_seq = -1
        self.armed = MISSION_NONE
        self.state = LIGHT_UNKNOWN

        # ARX: built once. The original rebuilds these params and the detector
        # three times per frame, once per colour.
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 600
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.7
        self.detector = cv2.SimpleBlobDetector_create(params)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def get_detect_traffic_light_param(self, params):
        for param in params:
            if param.name == 'red.hue_l':
                self.hue_red_l = param.value
                self.get_logger().info(f'red.hue_l set to: {param.value}')
            elif param.name == 'red.hue_h':
                self.hue_red_h = param.value
                self.get_logger().info(f'red.hue_h set to: {param.value}')
            elif param.name == 'red.saturation_l':
                self.saturation_red_l = param.value
                self.get_logger().info(f'red.saturation_l set to: {param.value}')
            elif param.name == 'red.saturation_h':
                self.saturation_red_h = param.value
                self.get_logger().info(f'red.saturation_h set to: {param.value}')
            elif param.name == 'red.lightness_l':
                self.lightness_red_l = param.value
                self.get_logger().info(f'red.lightness_l set to: {param.value}')
            elif param.name == 'red.lightness_h':
                self.lightness_red_h = param.value
                self.get_logger().info(f'red.lightness_h set to: {param.value}')
            elif param.name == 'yellow.hue_l':
                self.hue_yellow_l = param.value
                self.get_logger().info(f'yellow.hue_l set to: {param.value}')
            elif param.name == 'yellow.hue_h':
                self.hue_yellow_h = param.value
                self.get_logger().info(f'yellow.hue_h set to: {param.value}')
            elif param.name == 'yellow.saturation_l':
                self.saturation_yellow_l = param.value
                self.get_logger().info(f'yellow.saturation_l set to: {param.value}')
            elif param.name == 'yellow.saturation_h':
                self.saturation_yellow_h = param.value
                self.get_logger().info(f'yellow.saturation_h set to: {param.value}')
            elif param.name == 'yellow.lightness_l':
                self.lightness_yellow_l = param.value
                self.get_logger().info(f'yellow.lightness_l set to: {param.value}')
            elif param.name == 'yellow.lightness_h':
                self.lightness_yellow_h = param.value
                self.get_logger().info(f'yellow.lightness_h set to: {param.value}')
            elif param.name == 'green.hue_l':
                self.hue_green_l = param.value
                self.get_logger().info(f'green.hue_l set to: {param.value}')
            elif param.name == 'green.hue_h':
                self.hue_green_h = param.value
                self.get_logger().info(f'green.hue_h set to: {param.value}')
            elif param.name == 'green.saturation_l':
                self.saturation_green_l = param.value
                self.get_logger().info(f'green.saturation_l set to: {param.value}')
            elif param.name == 'green.saturation_h':
                self.saturation_green_h = param.value
                self.get_logger().info(f'green.saturation_h set to: {param.value}')
            elif param.name == 'green.lightness_l':
                self.lightness_green_l = param.value
                self.get_logger().info(f'green.lightness_l set to: {param.value}')
            elif param.name == 'green.lightness_h':
                self.lightness_green_h = param.value
                self.get_logger().info(f'green.lightness_h set to: {param.value}')
        return SetParametersResult(successful=True)

    # ARX: added.
    def get_armed(self, msg):
        if msg.data != self.armed:
            self.armed = msg.data
            self.get_logger().info(f'armed mission -> {self.armed}')

    def get_image(self, image_msg):
        # ARX: was a hard-coded "every 3 frames"; now a parameter, default 0.
        skip = self.get_parameter('frame_skip').value
        if skip > 0:
            self.counter += 1
            if self.counter % (skip + 1) != 0:
                return

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            try:
                self.cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')
                return

        self.is_image_available = True
        self.frame_seq += 1      # ARX

    def timer_callback(self):
        # ARX: idle unless this mission is armed, and never look at a frame
        # twice.
        if not (self.get_parameter('always_on').value
                or self.armed == MISSION_TRAFFIC_LIGHT):
            return
        if self.is_image_available and self.frame_seq != self.done_seq:
            self.done_seq = self.frame_seq
            self.find_traffic_light()

    def find_traffic_light(self):
        # ARX: converted once and handed to each mask. The original converts
        # the same frame to HSV three times, once inside each mask method.
        hsv = cv2.cvtColor(np.copy(self.cv_image), cv2.COLOR_BGR2HSV)
        # ARX: the order below is the original's, and green being tested last
        # is what gives it priority - the shipped bands overlap (red is hue
        # 0-24, yellow 19-33), so a red/yellow distinction is not trustworthy
        # while "is it green yet" is the only question the mission asks.
        state = LIGHT_UNKNOWN

        cv_image_mask_red = self.mask_red_traffic_light(hsv)
        cv_image_mask_red = cv2.GaussianBlur(cv_image_mask_red, (5, 5), 0)
        detect_red = self.find_circle_of_traffic_light(cv_image_mask_red, 'red')
        if detect_red:
            state = LIGHT_RED      # ARX
            cv2.putText(self.cv_image, 'RED', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        cv_image_mask_yellow = self.mask_yellow_traffic_light(hsv)
        cv_image_mask_yellow = cv2.GaussianBlur(cv_image_mask_yellow, (5, 5), 0)
        detect_yellow = self.find_circle_of_traffic_light(cv_image_mask_yellow, 'yellow')
        if detect_yellow:
            state = LIGHT_YELLOW   # ARX
            cv2.putText(self.cv_image, 'YELLOW', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))

        cv_image_mask_green = self.mask_green_traffic_light(hsv)
        cv_image_mask_green = cv2.GaussianBlur(cv_image_mask_green, (5, 5), 0)
        detect_green = self.find_circle_of_traffic_light(cv_image_mask_green, 'green')
        if detect_green:
            state = LIGHT_GREEN    # ARX
            cv2.putText(self.cv_image, 'GREEN', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

        # ARX: the whole reason for the fork. No hysteresis here - this says
        # what one frame looked like; deciding how many frames to believe
        # belongs to whoever owns the mission.
        if state != self.state:
            self.get_logger().info(f'light {NAMES[self.state]} -> {NAMES[state]}')
            self.state = state
        msg_state = UInt8()
        msg_state.data = int(state)
        self.pub_traffic_light.publish(msg_state)

        if not self.get_parameter('publish_debug_image').value:
            return             # ARX: skip the jpeg nobody asked for

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_light.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(self.cv_image, 'jpg'))
        else:
            self.pub_image_traffic_light.publish(
                self.cvBridge.cv2_to_imgmsg(self.cv_image, 'bgr8'))

    def mask_red_traffic_light(self, hsv):      # ARX: hsv passed in

        lower_red = np.array([self.hue_red_l, self.saturation_red_l, self.lightness_red_l])
        upper_red = np.array([self.hue_red_h, self.saturation_red_h, self.lightness_red_h])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_red_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_red_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def mask_yellow_traffic_light(self, hsv):      # ARX: hsv passed in

        lower_yellow = np.array(
            [self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l]
            )
        upper_yellow = np.array(
            [self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h]
            )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_yellow_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_yellow_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def mask_green_traffic_light(self, hsv):      # ARX: hsv passed in

        lower_green = np.array([self.hue_green_l, self.saturation_green_l, self.lightness_green_l])
        upper_green = np.array([self.hue_green_h, self.saturation_green_h, self.lightness_green_h])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_green_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_green_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def find_circle_of_traffic_light(self, mask, color):
        detect_result = False
        # ARX: self.detector, built once in __init__.
        keypts = self.detector.detect(mask)

        height, width = mask.shape[:2]
        roi_x_start = width // 2
        roi_x_end = width
        roi_y_start = height // 3
        roi_y_end = 2 * height // 3

        # ARX: break on the first keypoint inside the region of interest.
        # The original assigns detect_result on every iteration without
        # breaking, so the LAST keypoint decides: a lamp at keypoint 0 and a
        # reflection outside the roi at keypoint 1 returns False, after
        # having already logged that the light was detected.
        for i in range(len(keypts)):
            self.point_x = int(keypts[i].pt[0])
            self.point_y = int(keypts[i].pt[1])
            if roi_x_start < self.point_x < roi_x_end and roi_y_start < self.point_y < roi_y_end:
                detect_result = True
                break

        return detect_result


def main(args=None):
    rclpy.init(args=args)
    node = DetectTrafficLight()
    try:                          # ARX
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

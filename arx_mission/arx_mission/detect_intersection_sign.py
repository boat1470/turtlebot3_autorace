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
# Author: Leon Jung, Gilbert, Ashe Kim, Jun
#
# ARX: forked from turtlebot3_autorace_detect/detect_intersection_sign.py.
# Same file name, same class, same method names, same variable names, so a
# diff against the original shows only what changed. Every change is marked
# # ARX. Six of them:
#
#   1. The intersection sign is gone. Only left and right are matched.
#   2. An armed gate on /arx/armed_mission.
#   3. Left and right are now mutually exclusive - one publish per frame.
#   4. findHomography returning None, and frames with no features, no longer
#      kill the node.
#   5. MIN_MATCH_COUNT and MIN_MSE_DECISION are parameters.
#   6. frame_skip and publish_debug_image are parameters.
#
# Why the intersection sign went: it has 67 descriptors against left's 32 and
# right's 56, so it was both the slowest of the three to match and the most
# willing to match something that was not it. Nothing downstream ever needed
# it - the question this mission asks is which way to turn - and dropping it
# takes a third off the matching work on every frame, which is the most
# expensive thing this stack does on a Raspberry Pi 4.

from enum import Enum
import os

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8

# ARX: must match mission_control.py.
MISSION_INTERSECTION = 2


class DetectSign(Node):

    def __init__(self):
        super().__init__('detect_sign')

        # ARX: the original has no parameters at all - the two thresholds are
        # local variables inside the callback and the frame skip is a literal.
        self.declare_parameter('sign.min_match_count', 6)
        self.declare_parameter('sign.max_mse', 70000.0)
        # RANSAC inliers the homography must survive on.
        #
        # The original had no such check, and neither did the first version
        # of this fork: inliers were computed and used only to score left
        # against right. A frame can therefore pass min_match_count and
        # max_mse with every single match thrown out as an outlier, which
        # means the matches were scattered across the picture rather than
        # describing one rigid sign - a false positive by construction.
        #
        # This is not hypothetical. A run through the junction reported
        # "Detect right sign (0 inliers, score 0.000)" three times in a row
        # and outvoted the one report that had real support behind it
        # ("Detect left sign (6 inliers)"), on a course whose only arrow
        # board is the left one.
        #
        # Four is the floor rather than a threshold to tune: a homography is
        # fitted from four points, so RANSAC returning fewer than four
        # inliers means it found no consistent model at all.
        self.declare_parameter('sign.min_inliers', 4)
        self.declare_parameter('sign.frame_skip', 3)
        self.declare_parameter('sign.publish_debug_image', True)
        self.declare_parameter('sign.always_on', False)
        # ARX: which reference images to match against. The ones shipped in
        # turtlebot3_autorace_detect score zero RANSAC inliers on this
        # simulator's board at every distance measured - see
        # arx_mission/image/README.md - so the default is our own pair, cut
        # from the robot's own camera. Point this back at
        # turtlebot3_autorace_detect, or at a directory of images taken on
        # practice day, without touching the code.
        self.declare_parameter('sign.image_dir', '')

        self.sub_image_type = 'raw'  # you can choose image type 'compressed', 'raw'
        self.pub_image_type = 'compressed'  # you can choose image type 'compressed', 'raw'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage,
                '/detect/image_input/compressed',
                self.cbFindTrafficSign,
                10
            )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image,
                '/detect/image_input',
                self.cbFindTrafficSign,
                10
            )

        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign = self.create_publisher(
                CompressedImage,
                '/detect/image_output/compressed', 10
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign = self.create_publisher(
                Image, '/detect/image_output', 10
            )

        # ARX: the armed gate. SIFT over the whole frame is the most
        # expensive operation in this stack, and on the competition Pi there
        # will be five other detectors wanting the same CPU. Nothing reads a
        # detector that is not armed, so an unarmed one does not touch the
        # image at all.
        self.armed = 0
        self.create_subscription(UInt8, '/arx/armed_mission', self.cbArmedMission, 1)

        self.cvBridge = CvBridge()
        # ARX: kept exactly as the original wrote it even though
        # .intersection is now unused, because this line is what makes
        # left = 2 and right = 3. The values on /detect/traffic_sign are
        # unchanged, so the example's own tooling still reads them.
        self.TrafficSign = Enum('TrafficSign', 'intersection left right')
        self.counter = 1

        self.fnPreproc()

        self.get_logger().info('DetectSign Node Initialized')

    def cbArmedMission(self, msg):
        """Remember which mission the sequencer says is running (ARX)."""
        if msg.data == self.armed:
            return
        self.armed = msg.data
        self.get_logger().info(f'armed mission -> {self.armed}')

    def fnIsArmed(self):
        """Whether this frame should be looked at at all (ARX)."""
        if self.get_parameter('sign.always_on').value:
            return True
        return self.armed == MISSION_INTERSECTION

    def fnPreproc(self):
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()

        dir_path = self.get_parameter('sign.image_dir').value or os.path.join(
            get_package_share_directory('arx_mission'), 'image')  # ARX

        # ARX: intersection.png is no longer loaded.
        self.img_left = cv2.imread(dir_path + '/left.png', 0)
        self.img_right = cv2.imread(dir_path + '/right.png', 0)
        if any(img is None for img in (self.img_left, self.img_right)):
            raise FileNotFoundError(
                f'Reference sign image missing under {dir_path}'
            )

        self.kp_left, self.des_left = self.sift.detectAndCompute(self.img_left, None)
        self.kp_right, self.des_right = self.sift.detectAndCompute(self.img_right, None)

        self.get_logger().info(
            f'references from {dir_path}: left {len(self.des_left)} descriptors, '
            f'right {len(self.des_right)}'
        )

        FLANN_INDEX_KDTREE = 0
        index_params = {
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5
        }

        search_params = {
            'checks': 50
        }

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        squared_diff = (arr1 - arr2) ** 2
        total_sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1]  # cv_image_input and 2 should have same shape
        err = total_sum / num_all
        return err

    def fnMatchSign(self, kp1, des1, kp_ref, des_ref):
        """Match one reference sign and score it (ARX).

        Returns None when the sign is not there, otherwise the pieces the
        caller needs to publish and to draw: the good matches, the RANSAC
        inlier mask, and a score.

        The score is inliers divided by the number of descriptors in the
        reference image, not the raw count. left.png has 32 descriptors and
        right.png has 56, so comparing raw counts between them would lean
        towards right on every frame no matter what the camera is looking at.
        """
        min_match_count = self.get_parameter('sign.min_match_count').value
        max_mse = self.get_parameter('sign.max_mse').value

        matches = self.flann.knnMatch(des1, des_ref, k=2)

        good = []
        for pair in matches:
            # ARX: the original unpacked `for m, n in matches`. knnMatch
            # returns a shorter pair when the reference has few descriptors,
            # and that unpack raises ValueError when it does.
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) < min_match_count:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # ARX: RANSAC returns (None, None) when it cannot fit a model. The
        # original called mask.ravel() unconditionally, which raised
        # AttributeError inside the subscription callback and took the node
        # down - silently, in the middle of the course.
        if mask is None:
            return None

        mse = self.fnCalcMSE(src_pts, dst_pts)
        if mse >= max_mse:
            return None

        inliers = int(mask.sum())
        if inliers < self.get_parameter('sign.min_inliers').value:
            return None
        return {
            'good': good,
            'mask': mask.ravel().tolist(),
            'score': inliers / len(des_ref),
            'inliers': inliers,
            'mse': mse,
        }

    def cbFindTrafficSign(self, image_msg):
        # ARX: an unarmed detector does no work at all.
        if not self.fnIsArmed():
            return

        # drop the frame to 1/5 (6fps) because of the processing speed.
        # This is up to your computer's operating power.
        # ARX: a parameter now, and 0 means look at every frame.
        frame_skip = self.get_parameter('sign.frame_skip').value
        if frame_skip > 1:
            if self.counter % frame_skip != 0:
                self.counter += 1
                return
            else:
                self.counter = 1

        if self.sub_image_type == 'compressed':
            # converting compressed image to opencv image
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)

        # ARX: a frame with no features at all gives des1 = None, and
        # knnMatch raises on it. A frame with one feature cannot produce the
        # k=2 pairs the ratio test needs.
        if des1 is None or len(des1) < 2:
            self.fnPublishImage(cv_image_input)
            return

        match_left = self.fnMatchSign(kp1, des1, self.kp_left, self.des_left)
        match_right = self.fnMatchSign(kp1, des1, self.kp_right, self.des_right)

        # ARX: this is the change that matters most. The original tested left
        # and right in two independent `if` blocks, so a frame that matched
        # both published 2 and then 3 back to back - and anything keeping the
        # latest value read that as "right", every time. Picking one winner
        # means one publish per frame and no way to be told the wrong way
        # round by message ordering.
        if match_left and match_right:
            self.get_logger().info(
                f'both arrows matched - left {match_left["score"]:.3f} '
                f'right {match_right["score"]:.3f}'
            )
            best = match_left if match_left['score'] >= match_right['score'] else match_right
            is_left = best is match_left
        elif match_left:
            best, is_left = match_left, True
        elif match_right:
            best, is_left = match_right, False
        else:
            self.fnPublishImage(cv_image_input)
            return

        msg_sign = UInt8()
        msg_sign.data = (
            self.TrafficSign.left.value if is_left else self.TrafficSign.right.value
        )
        self.pub_traffic_sign.publish(msg_sign)
        self.get_logger().info(
            f'Detect {"left" if is_left else "right"} sign '
            f'({best["inliers"]} inliers, score {best["score"]:.3f}, '
            f'mse {best["mse"]:.0f})'
        )

        self.fnPublishImage(
            cv_image_input,
            kp1,
            best,
            self.img_left if is_left else self.img_right,
            self.kp_left if is_left else self.kp_right,
        )

    def fnPublishImage(self, cv_image_input, kp1=None, match=None, img_ref=None, kp_ref=None):
        """Publish the debug image, with the matches drawn on if there are any (ARX).

        The original inlined four near-identical copies of this, selected by
        an image_out_num that had to be kept in step with the branches above
        it by hand.
        """
        if not self.get_parameter('sign.publish_debug_image').value:
            return

        if match is None:
            final = cv_image_input
        else:
            draw_params = {
                'matchColor': (255, 0, 0),  # draw matches in green color
                'singlePointColor': None,
                'matchesMask': match['mask'],  # draw only inliers
                'flags': 2
            }
            final = cv2.drawMatches(
                cv_image_input,
                kp1,
                img_ref,
                kp_ref,
                match['good'],
                None,
                **draw_params
            )

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg')
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign.publish(
                self.cvBridge.cv2_to_imgmsg(final, 'bgr8')
            )


def main(args=None):
    rclpy.init(args=args)
    node = DetectSign()
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

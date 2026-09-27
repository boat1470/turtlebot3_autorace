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
# ARX: grew out of a fork of
# turtlebot3_autorace_detect/detect_intersection_sign.py, and no longer
# keeps its shape, so the name changed with it.
#
# The example ships five sign detectors - intersection, construction,
# parking, level_crossing, tunnel - and they are the same file five times
# over. Only four things differ between them: which reference images to
# load, what number to publish, how many matches to insist on, and the class
# name. Forking each one in turn would have meant five near-identical copies
# to keep in step.
#
# This is one node that reads that list from parameters instead. Adding the
# parking sign is a yaml entry, not a file.
#
# What it keeps from the fork:
#
#   1. An armed gate on /arx/armed_mission. Which signs are even considered
#      depends on which mission is running, so the node does no work at all
#      between missions - SIFT over a whole frame is the most expensive
#      thing in this stack and there is a Raspberry Pi 4 to share.
#   2. One publish per frame. The example tested each sign in its own `if`,
#      so a frame matching two published both, and anything keeping the
#      latest value read the second one - which for the arrows meant reading
#      "right" every time both matched.
#   3. Scores normalised by the reference's descriptor count, because
#      comparing raw match counts between a 32-descriptor image and a
#      56-descriptor one leans one way whatever the camera is looking at.
#   4. Guards for findHomography returning None, frames with no features,
#      and knnMatch pairs shorter than two - all of which killed the node.
#   5. A minimum inlier count. A match with no inliers at all is not a
#      detection: it means the matches were scattered rather than describing
#      one rigid sign. Without this the junction was called "right" on a
#      left board three times in a row.
#   6. Thresholds and the frame skip as parameters.

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


class DetectSign(Node):

    def __init__(self):
        super().__init__('detect_sign')

        # Which signs exist, and for each one: the mission that has to be
        # armed before it is looked for, the number to publish when it is
        # seen, and the file to match against.
        #
        #   sign:
        #     names: [left, right, construction]
        #     left:
        #       mission: 2
        #       value: 2
        #       image: left.png
        #
        # Signs sharing a mission are alternatives - the best scoring one
        # wins and the others are dropped. That is what stops a frame
        # matching both arrows from publishing both.
        self.declare_parameter('sign.names', ['left', 'right'])

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
        # ARX: the example kept the published numbers in
        # Enum('TrafficSign', 'intersection left right'), which fixed both
        # the set of signs and their values in code. They are parameters now
        # - sign.<name>.value - and the defaults keep the numbers the
        # example used, so its own tooling still reads them.
        self.counter = 1

        self.fnPreproc()

        self.get_logger().info('DetectSign Node Initialized')

    def cbArmedMission(self, msg):
        """Remember which mission the sequencer says is running (ARX)."""
        if msg.data == self.armed:
            return
        self.armed = msg.data
        self.get_logger().info(f'armed mission -> {self.armed}')

    def fnCandidates(self):
        """The signs worth looking for right now."""
        if self.get_parameter('sign.always_on').value:
            return list(self.signs.values())
        return [s for s in self.signs.values() if s['mission'] == self.armed]

    def fnPreproc(self):
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()

        # ARX: the reference set comes from parameters. Empty means this
        # package's own image directory, whose README records why the ones
        # in turtlebot3_autorace_detect are not used - they score zero
        # RANSAC inliers against this simulator's boards at every distance
        # measured.
        dir_path = self.get_parameter('sign.image_dir').value or os.path.join(
            get_package_share_directory('arx_mission'), 'image')

        self.signs = {}
        for name in self.get_parameter('sign.names').value:
            for key, default in (('mission', 0), ('value', 0), ('image', f'{name}.png')):
                self.declare_parameter(f'sign.{name}.{key}', default)
            image = self.get_parameter(f'sign.{name}.image').value
            img = cv2.imread(os.path.join(dir_path, image), 0)
            if img is None:
                raise FileNotFoundError(
                    f'reference image for "{name}" missing: '
                    f'{os.path.join(dir_path, image)}')
            kp, des = self.sift.detectAndCompute(img, None)
            if des is None:
                raise ValueError(f'no SIFT features in {image}')
            self.signs[name] = {
                'name': name,
                'mission': self.get_parameter(f'sign.{name}.mission').value,
                'value': self.get_parameter(f'sign.{name}.value').value,
                'img': img, 'kp': kp, 'des': des,
            }
            self.get_logger().info(
                f'  {name}: {image} {img.shape[1]}x{img.shape[0]} '
                f'{len(des)} descriptors, publishes '
                f'{self.signs[name]["value"]} while mission '
                f'{self.signs[name]["mission"]} is armed')

        if not self.signs:
            raise ValueError('sign.names is empty - nothing to look for')

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

    def fnMatchSign(self, kp1, des1, sign):
        """Match one reference sign and score it (ARX).

        Returns None when the sign is not there, otherwise the pieces the
        caller needs to publish and to draw: the good matches, the RANSAC
        inlier mask, and a score.

        The score is inliers divided by the number of descriptors in the
        reference image, not the raw count. left.png has 32 descriptors and
        right.png has 56, so comparing raw counts between them would lean
        towards right on every frame no matter what the camera is looking at.
        """
        kp_ref, des_ref = sign['kp'], sign['des']
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
            'sign': sign,
            'good': good,
            'mask': mask.ravel().tolist(),
            'score': inliers / len(des_ref),
            'inliers': inliers,
            'mse': mse,
        }

    def cbFindTrafficSign(self, image_msg):
        # ARX: with nothing to look for, no work is done at all.
        candidates = self.fnCandidates()
        if not candidates:
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

        hits = [m for m in (self.fnMatchSign(kp1, des1, s) for s in candidates)
                if m is not None]

        # ARX: one winner per frame. The example tested each sign in its own
        # `if` and published every one that matched, so anything keeping the
        # latest value saw whichever happened to be tested last.
        if not hits:
            self.fnPublishImage(cv_image_input)
            return
        if len(hits) > 1:
            self.get_logger().info(
                'several signs matched - '
                + ', '.join(f'{h["sign"]["name"]} {h["score"]:.3f}' for h in hits))
        best = max(hits, key=lambda h: h['score'])

        msg_sign = UInt8()
        msg_sign.data = int(best['sign']['value'])
        self.pub_traffic_sign.publish(msg_sign)
        self.get_logger().info(
            f'Detect {best["sign"]["name"]} sign '
            f'({best["inliers"]} inliers, score {best["score"]:.3f}, '
            f'mse {best["mse"]:.0f})')

        self.fnPublishImage(cv_image_input, kp1, best,
                            best['sign']['img'], best['sign']['kp'])

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

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
        # ARX: the arrow reader - see fnReadArrow. Read every frame, so all of
        # these can be set live with `ros2 param set`.
        for key, default in (('hue_l', 95), ('hue_h', 130), ('sat_l', 80), ('val_l', 40),
                             ('sat_h', 255), ('val_h', 255),
                             ('min_area', 40.0), ('max_area', 4000.0), ('min_round', 0.75),
                             ('min_fill', 0.30), ('min_white', 6), ('white_delta', 25.0),
                             ('min_offset_px', 1.5), ('roi_y_min', 0.0), ('roi_y_max', 1.0)):
            self.declare_parameter(f'sign.arrow.{key}', default)

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
        # ARX: the arrow reader's blue mask, with the discs it weighed boxed -
        # where the colour thresholds are set from. Remapped to
        # /detect/image_traffic_sign_mask/compressed by the launch file.
        self.pub_image_arrow_mask = self.create_publisher(
            CompressedImage, '/detect/image_output_sub1/compressed', 10)
        self.arrow_mask = None

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
            # ARX: min_width_px - how wide the board has to look before it
            # counts, i.e. how close the robot has to be. 0 is any size.
            # max_mse, min_inliers - this sign's own, -1 for the shared one.
            # max_width_px - wider than this is not the board (a fit spread
            # over half the picture); 0 is no limit.
            # method - sift, or arrow for a blue disc whose white arrow is read
            # by colour (fnReadArrow); arrow - which way that disc's arrow
            # points when it counts as this sign.
            for key, default in (('mission', 0), ('value', 0), ('image', f'{name}.png'),
                                 ('min_width_px', 0.0), ('max_mse', -1.0),
                                 ('min_inliers', -1), ('max_width_px', 0.0),
                                 ('method', 'sift'), ('arrow', name)):
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
                'min_width_px': float(self.get_parameter(f'sign.{name}.min_width_px').value),
                'max_mse': float(self.get_parameter(f'sign.{name}.max_mse').value),
                'min_inliers': int(self.get_parameter(f'sign.{name}.min_inliers').value),
                'max_width_px': float(self.get_parameter(f'sign.{name}.max_width_px').value),
                'method': self.get_parameter(f'sign.{name}.method').value,
                'arrow': self.get_parameter(f'sign.{name}.arrow').value,
                'img': img, 'kp': kp, 'des': des,
            }
            how = ('read by colour, arrow ' + self.signs[name]['arrow']
                   if self.signs[name]['method'] == 'arrow'
                   else f'{len(des)} descriptors')
            self.get_logger().info(
                f'  {name}: {image} {img.shape[1]}x{img.shape[0]} '
                f'{how}, publishes '
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
        # ARX: per sign, 0 to switch it off. It compares the board's pixel
        # coordinates in the frame against those in the reference image, so
        # what it really measures is how far the board sits from the frame's
        # top-left corner and how much bigger it looks - a close, large board
        # fails it with every match an inlier (132 px wide at (200, 100):
        # 11/11 inliers, mse 81940). A sign that must be seen big cannot use it.
        max_mse = sign['max_mse']
        if max_mse < 0:
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
        if max_mse > 0 and mse >= max_mse:
            return None

        inliers = int(mask.sum())
        min_inliers = sign['min_inliers']
        if min_inliers < 0:
            min_inliers = self.get_parameter('sign.min_inliers').value
        if inliers < min_inliers:
            return None

        # ARX: where the board is in the picture, and so how big. M maps the
        # camera onto the reference, so its inverse carries the reference's
        # corners back into the frame.
        h, w = sign['img'].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        try:
            box = cv2.perspectiveTransform(corners, np.linalg.inv(M))
        except np.linalg.LinAlgError:
            return None
        xs = box[:, 0, 0]
        width = float(xs.max() - xs.min())
        # A fit that mirrors or folds the board over is not a board. A real
        # view keeps the corners going round the same way as the reference,
        # which OpenCV's oriented area gives as positive.
        if not np.isfinite(width) or cv2.contourArea(box, oriented=True) <= 0:
            return None
        if sign['max_width_px'] > 0 and width > sign['max_width_px']:
            return None
        return {
            'sign': sign,
            'good': good,
            'mask': mask.ravel().tolist(),
            'score': inliers / len(des_ref),
            'inliers': inliers,
            'mse': mse,
            'width': width,
            'box': box,
            # Matched, but not yet as big as it has to be. Not a detection -
            # kept only so the debug image can show how big it is.
            'small': width < sign['min_width_px'],
        }

    def fnReadArrow(self, image):
        """Which way the arrow on a blue disc points, read by colour (ARX).

        Returns (read, rejects): read is None or {'dir', 'box', 'head',
        'stem', 'offset', 'white'}; rejects is [(box, why)] for the discs
        that did not make it, for the debug image.

        SIFT needs the board big: at the distance the robot actually sees the
        junction board it is a 15 px disc with a 5 px arrow, where it finds
        nothing to match (2026-10-23 live frame: 0 inliers with the new
        references, 4 on the wall with the old). Colour still finds the disc
        there, and the white inside it still has its head on one side of its
        stem: tested on that frame (15x17 px) and on frames/left_frame.png,
        right_frame.png, all three read right.

        1. blue by HSV, closed, as connected blobs, biggest first
        2. a disc: area in range, near round (min side / max side) - the arm
           that is down is a disc seen edge on, 15x9
        3. white inside it: the holes in the blob, and pixels brighter than
           the disc's mean by white_delta; some in each half
        4. x of the white in the top half (the head) against the bottom half
           (the stem): head left of stem is left
        """
        g = lambda k: self.get_parameter(f'sign.arrow.{k}').value  # noqa: E731
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([g('hue_l'), g('sat_l'), g('val_l')]),
                           np.array([g('hue_h'), g('sat_h'), g('val_h')]))
        # The band searched: rows outside roi_y_min..roi_y_max (fractions of
        # the height, from the top) are blanked before anything is looked
        # for, so the mask image shows exactly what was searched.
        rows = image.shape[0]
        y0 = int(round(min(max(g('roi_y_min'), 0.0), 1.0) * rows))
        y1 = int(round(min(max(g('roi_y_max'), 0.0), 1.0) * rows))
        blue[:y0] = 0
        blue[y1:] = 0
        self.roi_rows = (y0, y1)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.arrow_mask = blue
        n, lab, st, _ = cv2.connectedComponentsWithStats(blue)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rejects = []
        for i in sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA]):
            x, y, w, h, area = (int(v) for v in st[i])
            box = (x, y, w, h)
            if area < g('min_area'):
                break                       # sorted, so the rest are smaller
            if area > g('max_area'):
                continue
            rnd = min(w, h) / max(w, h)
            if rnd < g('min_round'):
                rejects.append((box, f'round {rnd:.2f}'))
                continue
            # A disc fills its box (~0.5 with the arrow cut out of it); a thin
            # curved sliver does not, though its box can be square. Run
            # 1023_2105 read left off one at the frame's right edge, fill ~0.15,
            # while the real discs were 0.43-0.64.
            fill = area / float(w * h)
            if fill < g('min_fill'):
                rejects.append((box, f'fill {fill:.2f}'))
                continue
            disc = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            filled = disc.copy()
            cnts, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled, cnts, -1, 255, -1)
            inside = filled > 0
            crop = gray[y:y + h, x:x + w]
            white = (inside & (disc == 0)) | (
                inside & (crop > crop[inside].mean() + g('white_delta')))
            ys, xs = np.nonzero(white)
            top = ys < h / 2.0
            if len(xs) < g('min_white') or not top.any() or top.all():
                rejects.append((box, f'white {len(xs)}'))
                continue
            hx, sx = xs[top].mean(), xs[~top].mean()
            offset = float(hx - sx)
            if abs(offset) < g('min_offset_px'):
                rejects.append((box, f'off {offset:+.1f}'))
                continue
            return ({'dir': 'left' if offset < 0 else 'right', 'box': box,
                     'head': (x + hx, y + ys[top].mean()),
                     'stem': (x + sx, y + ys[~top].mean()),
                     'offset': offset, 'white': len(xs)}, rejects)
        return None, rejects

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

        # ARX: the signs read by colour first. One of them found ends the
        # frame, as a SIFT winner does; with none of the SIFT kind armed (the
        # junction) SIFT does not run at all.
        arrow_signs = [s for s in candidates if s['method'] == 'arrow']
        candidates = [s for s in candidates if s['method'] != 'arrow']
        if arrow_signs:
            read, rejects = self.fnReadArrow(cv_image_input)
            sign = None if read is None else next(
                (s for s in arrow_signs if s['arrow'] == read['dir']), None)
            if sign is not None:
                msg_sign = UInt8()
                msg_sign.data = int(sign['value'])
                self.pub_traffic_sign.publish(msg_sign)
                x, y, w, h = read['box']
                self.get_logger().info(
                    f'Detect {sign["name"]} sign (arrow, disc {w}x{h} px, '
                    f'head-stem {read["offset"]:+.1f} px, {read["white"]} white px)')
                self.fnPublishImage(cv_image_input, arrow=read, rejects=rejects)
                return
            if not candidates:
                self.fnPublishImage(cv_image_input, rejects=rejects)
                return

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)

        # ARX: a frame with no features at all gives des1 = None, and
        # knnMatch raises on it. A frame with one feature cannot produce the
        # k=2 pairs the ratio test needs.
        if des1 is None or len(des1) < 2:
            self.fnPublishImage(cv_image_input)
            return

        matched = [m for m in (self.fnMatchSign(kp1, des1, s) for s in candidates)
                   if m is not None]
        hits = [m for m in matched if not m['small']]
        small = [m for m in matched if m['small']]

        # ARX: one winner per frame. The example tested each sign in its own
        # `if` and published every one that matched, so anything keeping the
        # latest value saw whichever happened to be tested last.
        if not hits:
            if small:
                # Drawn in red with its width, so min_width_px can be set by
                # watching the number grow as the robot comes in.
                self.fnPublishImage(cv_image_input,
                                    small=max(small, key=lambda h: h['width']))
            else:
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
            f'mse {best["mse"]:.0f}, width {best["width"]:.0f} px)')

        self.fnPublishImage(cv_image_input, kp1, best,
                            best['sign']['img'], best['sign']['kp'])

    def fnDrawBox(self, image, match, colour):
        """The board's outline and width in px - what min_width_px is set from (ARX).

        Yellow for a detection, red for a board matched but still too small.
        """
        image = image.copy()
        cv2.polylines(image, [np.int32(match['box'])], True, colour, 2)
        x, y = np.int32(match['box'][:, 0, :].min(axis=0))
        need = match['sign']['min_width_px']
        text = f'{match["sign"]["name"]} {match["width"]:.0f} px'
        if need > 0:
            text += f' / {need:.0f}'
        cv2.putText(image, text, (int(x), max(int(y) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return image

    def fnDrawArrow(self, image, read, rejects):
        """The arrow reader's view: discs it dropped and why, the one it read (ARX)."""
        image = image.copy()
        # The searched band, as two dashed lines - only where it cuts anything.
        y0, y1 = getattr(self, 'roi_rows', (0, image.shape[0]))
        for yy in (y0, y1 - 1):
            if 0 < yy < image.shape[0] - 1:
                for xx in range(0, image.shape[1], 8):
                    cv2.line(image, (xx, yy), (xx + 4, yy), (255, 0, 255), 1)
        for (x, y, w, h), why in rejects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (150, 150, 150), 1)
            cv2.putText(image, why, (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (150, 150, 150), 1)
        if read is not None:
            x, y, w, h = read['box']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.circle(image, tuple(int(v) for v in read['head']), 2, (0, 0, 255), -1)
            cv2.circle(image, tuple(int(v) for v in read['stem']), 2, (0, 255, 0), -1)
            cv2.putText(image, f'{read["dir"]} {read["offset"]:+.1f}', (x, max(y - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        elif not rejects:
            cv2.putText(image, 'no blue disc', (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (150, 150, 150), 1)
        return image

    def fnPublishImage(self, cv_image_input, kp1=None, match=None, img_ref=None, kp_ref=None,
                       small=None, arrow=None, rejects=None):
        """Publish the debug image, with the matches drawn on if there are any (ARX).

        The original inlined four near-identical copies of this, selected by
        an image_out_num that had to be kept in step with the branches above
        it by hand.
        """
        if not self.get_parameter('sign.publish_debug_image').value:
            return

        if rejects is not None:
            final = self.fnDrawArrow(cv_image_input, arrow, rejects)
            if self.arrow_mask is not None:
                self.pub_image_arrow_mask.publish(self.cvBridge.cv2_to_compressed_imgmsg(
                    self.fnDrawArrow(cv2.cvtColor(self.arrow_mask, cv2.COLOR_GRAY2BGR),
                                     arrow, rejects), 'jpg'))
        elif match is None:
            final = cv_image_input
            if small is not None:
                final = self.fnDrawBox(cv_image_input, small, (0, 0, 255))
        else:
            cv_image_input = self.fnDrawBox(cv_image_input, match, (0, 255, 255))
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

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

"""The real camera, for the robot rather than the simulator.

A fork of turtlebot3_bringup/launch/camera.launch.py. Run it instead of that
one:

    ros2 launch arx_camera_bringup camera.launch.py

Why a fork rather than an edit. The sensor mode in the original is hardcoded
in the parameters dict rather than exposed as an argument, so the one thing
that has to change cannot be changed from the command line - and the file
belongs to turtlebot3_bringup, so an edit there is lost on the next update
and invisible to the other machine.

ARX: three changes, all measured against this camera.

1. sensor_mode 1640:1232 -> 1296:972.

   1640x1232 is the binned mode of the IMX219, which is Camera Module v2.
   This robot carries an OV5647 - Camera Module v1 - which has no such mode,
   so libcamera quietly fell back to the full sensor and the log said so:

       Sensor mode configuration: 1640x1232-SGBRG10_CSI2P/RAW     requested
       configuring streams: (1) 2592x1944-SGBRG10_CSI2P/RAW       given
       Selected sensor format: 2592x1944-SGBRG10_1X10/RAW

   Reading all five megapixels caps the OV5647 at about 15 fps, and 13.3 Hz
   is what /camera/image_raw measured. 1296x972 is its 2x2 binned mode:
   around 42 fps, four pixels of light summed into one, and 4:3 like the
   320x240 output, so nothing is cropped off the top and bottom - which is
   where the road is.

   Not 1920x1080. It is faster than full resolution but 16:9, so it crops
   away the part of the frame the lane lines are in.

2. format exposed and defaulted.

   The original leaves it empty, and camera_ros then says:

       no pixel format selected, auto-selecting: "XRGB8888"

   which is four bytes per pixel and reaches ROS as bgra8. The simulator
   gives bgr8, so everything in this package was written against three
   channels.

   RGB888 here is libcamera's name, and libcamera names a format by the
   order its bytes sit in memory, which is the reverse of how ROS reads it.
   libcamera RGB888 is expected to arrive as ROS bgr8. CHECK IT rather than
   trust it, because getting it backwards swaps red and blue and every hue
   threshold in this package then quietly means something else:

       ros2 topic echo /camera/image_raw --once --field encoding

   bgr8 is right. rgb8 means use format:=BGR888 instead.

3. Exposure, white balance and sharpness exposed, all defaulting to the
   camera's own behaviour.

   The simulator has no auto-exposure and no auto-white-balance, so every
   HSV threshold in this package was tuned against a camera whose colour
   response does not move. On this one AwbEnable is on by default and will
   shift the hue of the lane lines with whatever else is in frame.

   They default to leaving the camera alone, because locking exposure
   without a measured value for it gives a black frame. Measure first, then
   pass them - and write the numbers down, because every threshold tuned
   afterwards is only valid under them.

       ros2 launch arx_camera_bringup camera.launch.py \\
           ae:=false awb:=false sharpness:=0.0 exposure_us:=8000 gain:=2.0

Frame rate is not set here. camera_ros tries and fails on its own -

    FrameDurationLimits: cannot set default scalar value '33333' on span
    control (extend: 2), default will be ignored

- because that control wants a (min, max) pair. Set it at runtime if it is
ever needed:

    ros2 param set /camera FrameDurationLimits "[33333, 33333]"

The intrinsic calibration file is named after the resolution AND the sensor
mode, so changing either invalidates it:

    ~/.ros/camera_info/ov5647__..._320x240_1296x972_SGBRG10_CSI2P_RAW.yaml

Settle this file before calibrating, not after.
"""

from ament_index_python.resources import has_resource

from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description import LaunchDescription
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def as_bool(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ('true', '1')


def as_camera(context):
    """The camera as an index if it reads as one, otherwise as a name (ARX).

    LaunchConfiguration.perform always returns a string, and camera_ros reads
    a string as the camera's NAME. Passing "0" that way asks for a camera
    called "0" and gets:

        >> cameras:
           0: ov5647 (/base/soc/i2c0mux/i2c@1/ov5647@36)
        ERROR: camera with name 0 does not exist

    The original launch file never hits this because it hands the
    LaunchConfiguration straight to launch_ros, which infers the type and
    turns "0" into an integer. Resolving it here loses that, so the
    conversion has to be done by hand.
    """
    value = LaunchConfiguration('camera').perform(context)
    return int(value) if value.lstrip('-').isdigit() else value


def camera(context, *unused_args, **unused_kwargs):
    """Build the camera node once the arguments have values (ARX).

    An OpaqueFunction because three of the controls are only passed when the
    user asks for them: a libcamera control that is set to a placeholder is
    not the same as one left alone, and ExposureTime in particular gives a
    black frame if it is set without a measured value.
    """
    params = {
        'camera': as_camera(context),
        'sensor_mode': LaunchConfiguration('sensor_mode').perform(context),
        'width': int(LaunchConfiguration('width').perform(context)),
        'height': int(LaunchConfiguration('height').perform(context)),
        'format': LaunchConfiguration('format').perform(context),
        'AeEnable': as_bool(context, 'ae'),
        'AwbEnable': as_bool(context, 'awb'),
    }

    # Negative means "do not touch it", which is not the same as zero -
    # Sharpness 0.0 is a real setting that turns sharpening off, and that is
    # what measuring focus wants.
    sharpness = float(LaunchConfiguration('sharpness').perform(context))
    if sharpness >= 0.0:
        params['Sharpness'] = sharpness

    # Only meaningful with ae:=false, and only after the right value has been
    # measured in the light the robot will actually run in.
    exposure = int(LaunchConfiguration('exposure_us').perform(context))
    if exposure > 0:
        params['ExposureTime'] = exposure

    gain = float(LaunchConfiguration('gain').perform(context))
    if gain > 0.0:
        params['AnalogueGain'] = gain

    nodes = [
        ComposableNode(
            package='camera_ros',
            plugin='camera::CameraNode',
            parameters=[params],
            extra_arguments=[{'use_intra_process_comms': True}],
        ),
    ]

    if has_resource('packages', 'image_view'):
        nodes.append(
            ComposableNode(
                package='image_view',
                plugin='image_view::ImageViewNode',
                remappings=[('/image', '/camera/image_raw')],
                extra_arguments=[{'use_intra_process_comms': True}],
                condition=IfCondition(LaunchConfiguration('use_image_view')),
            )
        )

    return [ComposableNodeContainer(
        name='camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=nodes,
    )]


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            'camera', default_value='0',
            description='Camera index, or its name. A number is taken as an '
                        'index - the node lists what it found at startup.'),
        # ARX: the one the original hardcodes. See the note at the top - the
        # value it hardcodes belongs to a different sensor.
        DeclareLaunchArgument(
            'sensor_mode', default_value='1296:972',
            description="The OV5647's 2x2 binned mode: full field of view, "
                        'about 42 fps. 2592:1944 is the full sensor at 15 '
                        'fps; 1920:1080 is 16:9 and crops away the road.'),
        DeclareLaunchArgument(
            'width', default_value='320',
            description='Output width. 320x240 is what the whole package is '
                        'written against, including the hardcoded trapezoid '
                        'in image_projection.py.'),
        DeclareLaunchArgument(
            'height', default_value='240',
            description='Output height.'),
        DeclareLaunchArgument(
            'format', default_value='RGB888',
            description="libcamera's name, whose byte order is the reverse "
                        'of how ROS reads it - this is expected to arrive as '
                        'bgr8. Check with `ros2 topic echo '
                        '/camera/image_raw --once --field encoding`; if it '
                        'says rgb8, use BGR888 instead.'),
        DeclareLaunchArgument(
            'ae', default_value='true',
            description='Auto exposure. Lock it off before tuning any hue '
                        'threshold, and pass exposure_us with it.'),
        DeclareLaunchArgument(
            'awb', default_value='true',
            description='Auto white balance. On by default and it moves the '
                        'hue of the lane lines with whatever is in frame; '
                        'the simulator has no such thing.'),
        DeclareLaunchArgument(
            'sharpness', default_value='-1.0',
            description='Negative leaves it alone. 0.0 turns sharpening off, '
                        'which is what measuring focus wants - sharpening '
                        'inflates any sharpness metric.'),
        DeclareLaunchArgument(
            'exposure_us', default_value='0',
            description='Microseconds. 0 leaves it alone. Only meaningful '
                        'with ae:=false, and only once measured.'),
        DeclareLaunchArgument(
            'gain', default_value='0.0',
            description='Analogue gain. 0 leaves it alone.'),
        DeclareLaunchArgument(
            'use_image_view', default_value='false',
            description='Also open image_view, if it is installed.'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=camera)])

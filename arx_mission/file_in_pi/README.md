# Staging for the robot

Packages that belong on the Pi and nowhere else. They are **not built on the
development machine** - they are copied to the robot and built there.

Two things keep them out of this machine's build, and both are deliberate:

1. This folder sits inside `arx_mission/`, and colcon stops descending once
   it has identified a directory as a package. A `package.xml` below one is
   never found.
2. `COLCON_IGNORE` says the same thing explicitly, so the invisibility does
   not depend on that behaviour staying the way it is - or on the folder
   staying where it is.

Checked: `colcon list --names-only` shows `arx_mission` and not
`arx_camera_bringup`.

## Copying it over

The robot keeps these in a workspace of their own, `~/arx_mission_pi`, apart
from the two it already had.

    rsync -av --delete \
        ~/turtlebot3_ws/src/turtlebot3_autorace/arx_mission/file_in_pi/arx_camera_bringup \
        iotcentral@192.168.0.101:~/arx_mission_pi/src/

Then on the Pi:

    cd ~/arx_mission_pi && colcon build --symlink-install

## Sourcing on the Pi, in this order

There are three workspaces there and they overlay, so the order is not a
matter of taste - a workspace can only see what was sourced before it.

    source /opt/ros/jazzy/setup.bash
    source ~/camera_ws/install/setup.bash       # camera_ros
    source ~/turtlebot3_ws/install/setup.bash   # turtlebot3_bringup, drivers
    source ~/arx_mission_pi/install/setup.bash  # this

`arx_camera_bringup` loads `camera_ros`'s composable node, so `camera_ws` has
to be underneath it. Get the order wrong and the container comes up and then
fails to find the component, which reads as a missing package rather than a
missing source line.

## What is in here

### `arx_camera_bringup`

`camera.launch.py` - a fork of `turtlebot3_bringup`'s, because that one
hardcodes a sensor mode belonging to a different camera. The reasoning and
the measurements are in the file's own docstring.

### `camera_info/`

The intrinsic calibration, measured 2026-10-04. A backup only: the live copy
is on the robot at

    ~/.ros/camera_info/ov5647__base_soc_i2c0mux_i2c_1_ov5647_36_320x240_1296x972_SGBRG10_CSI2P_RAW.yaml

`camera_ros` reads that path by itself on start and publishes what it finds on
`/camera/camera_info`, so nothing in a launch file refers to it. Put it back
with a plain copy if the card is ever reflashed:

    scp arx_mission/file_in_pi/camera_info/*.yaml \
        iotcentral@192.168.0.101:~/.ros/camera_info/

The file name is not decoration. `camera_ros` builds it from the sensor, the
I2C address, the output size and the sensor mode, and loads a file only if the
name matches exactly what it would generate now. Change `width`, `height` or
`sensor_mode` in `camera.launch.py` and this file stops being found - silently,
with `camera_info` dropping back to all zeros and rectification quietly doing
nothing. That is what it looked like before this was measured.

Measured with `cameracalibrator` on the dev machine, 196 images, board 8x6
interior corners. Pinhole, not fisheye: the lens is sold as 160 degrees but
that is the bare lens, and the 1/4 inch sensor sees about 112 degrees across.
Checked afterwards on a frame of the board - the bow in its rows and columns
fell from 2.27 px to 0.56 px, and the systematic pattern in it disappeared.

The raw images are kept outside git in `notes/calibrationdata_2026-10-04.tar.gz`,
so the fit can be redone - with the fisheye model, say - without setting the
board up again.

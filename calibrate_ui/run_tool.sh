#!/usr/bin/env bash
# Start one of the calibrate_ui tools with the ROS environment a terminal
# would have - for the desktop icons, which start with none of ~/.bashrc.
#
#     calibrate_ui/run_tool.sh launcher.py
#     calibrate_ui/run_tool.sh lane_debug.py --no-image
#
# The environment is the same lines ~/.bashrc uses for this robot. Output
# goes to ~/.ros/log/<tool>.desktop.log, so a window that never opens can
# still be explained.
set -e
source /opt/ros/jazzy/setup.bash
[ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export TURTLEBOT3_MODEL=burger_cam

here="$(cd "$(dirname "$0")" && pwd)"
tool="$1"; shift
mkdir -p ~/.ros/log
cd "$here/.."
exec python3 "$here/$tool" "$@" >> ~/.ros/log/"${tool%.py}".desktop.log 2>&1

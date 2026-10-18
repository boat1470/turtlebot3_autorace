#!/bin/bash
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0
#
# Put arx_ultrasonic_bringup on the Pi and build it there, so the launcher's
# "level ultrasonic" button (ros2 launch arx_ultrasonic_bringup
# ultrasonic.launch.py) has something to run. Run it again after changing the
# node or param/ultrasonic.yaml.
#
#     calibrate_ui/pi_sync.sh                       # iotcentral@192.168.0.101
#     calibrate_ui/pi_sync.sh user@host
#
# Goes into ~/arx_mission_pi/src next to arx_camera_bringup; the Pi's ~/.bashrc
# already sources ~/arx_mission_pi/install.

set -euo pipefail
HOST=${1:-iotcentral@192.168.0.101}
PKG=arx_ultrasonic_bringup
SRC="$(cd "$(dirname "$0")/.." && pwd)/$PKG"

echo "== copying $SRC -> $HOST:~/arx_mission_pi/src/$PKG"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
    "$SRC/" "$HOST:arx_mission_pi/src/$PKG/"

echo "== building on the Pi"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" \
    "bash -ic 'cd ~/arx_mission_pi && colcon build --packages-select $PKG 2>&1 | tail -3'"

echo "== check"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" \
    "bash -ic 'ros2 pkg executables $PKG'"
echo "done - the launcher's 'level ultrasonic' button can start it now"

#!/bin/bash
# Draw one run's graphs into graph/<run>/:  ./draw_run.sh 1101_2147
#   analysis.png  speed, turn rate, heading, lane error, lane_state over time
#   map_fit.png   path on docs/curve_map.png, odom pulled onto the lane
#   map_raw.png   path on docs/curve_map.png, odom as recorded
set -e
here="$(cd "$(dirname "$0")" && pwd)"
run="${1:?usage: $0 <run, e.g. 1101_2147>}"
bag="$HOME/runs/$run"
[ -f "$bag/metadata.yaml" ] || bag="$(ls "$bag"/*.mcap | head -1)"
mkdir -p "$here/$run"
python3 "$here/analyze.py" "$here/$run/analysis.png" "$bag" 2>&1 | grep -v WARN
python3 "$here/mapfit.py"  "$here/$run/map_fit.png"  "$bag" 2>&1 | grep -v WARN
python3 "$here/fullmap.py" "$here/$run/map_raw.png"  "$bag" 2>&1 | grep -v WARN || true
echo "-> $here/$run"

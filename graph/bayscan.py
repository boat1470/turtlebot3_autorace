# What the LiDAR saw when the robot read the bays, robot's frame, forward up.
#   python3 bayscan.py out.png <run>
import os, sys, math, numpy as np, rosbag2_py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import Log
out, run = sys.argv[1], sys.argv[2]
HALF, REACH, NEED = 30.0, 0.5, 3          # detect_obstacle's side_angle_deg, side_distance, side_min_points
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=os.path.expanduser('~/runs/' + run), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
r.set_filter(rosbag2_py.StorageFilter(topics=['/scan', '/rosout']))
scans, tb, why = [], None, ''
while r.has_next():
    t, raw, ts = r.read_next()
    if t == '/rosout':
        m = deserialize_message(raw, Log).msg
        if 'bays read' in m and tb is None: tb, why = ts * 1e-9, m
    else: scans.append((ts * 1e-9, deserialize_message(raw, LaserScan)))
s = min((x for x in scans if x[0] <= tb), key=lambda x: tb - x[0])[1]
a = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment
rr = np.array(s.ranges); ok = np.isfinite(rr) & (rr >= s.range_min) & (rr <= s.range_max)
deg = (np.degrees(a) + 180) % 360 - 180
# forward up, robot's left to the left: plot x = -y_robot, y = x_robot
px, py = -rr * np.sin(a), rr * np.cos(a)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
cols = {90: '#2a78d6', -90: '#eb6834'}
for c, name in [(90, 'left'), (-90, 'right')]:
    sel = ok & (rr <= REACH) & (abs(deg - c) <= HALF)
    n = int(sel.sum())
    # matplotlib angles: 0 = +x (right), CCW; robot angle c maps to 90 + c
    ax.add_patch(Wedge((0, 0), REACH, 90 + c - HALF, 90 + c + HALF, color=cols[c], alpha=0.12))
    ax.scatter(px[sel], py[sel], s=30, color=cols[c], zorder=3)
    lx = -0.42 * math.sin(math.radians(c))
    ax.text(lx, 0.40, f'{name}: {n} points\n{"TAKEN" if n >= NEED else "clear"}'
            + (f'\nnearest {rr[sel].min():.3f} m' if n else ''), ha='center', fontsize=11, color='#333')
far = ok & ~(((abs(deg - 90) <= HALF) | (abs(deg + 90) <= HALF)) & (rr <= REACH))
ax.scatter(px[far], py[far], s=6, color='#999', zorder=2, label='other returns')
ax.add_patch(Rectangle((-0.069, -0.089 + 0.032), 0.138, 0.178, fill=False, ec='k', lw=1.5))   # Burger body, LiDAR 32 mm behind centre
ax.annotate('', (0, 0.12), (0, 0.0), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(0, 0.13, 'forward', ha='center', fontsize=9)
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('m (robot frame, left is left)'); ax.set_ylabel('m (forward)')
ax.set_title(f'{run} - LiDAR scan as the bays were read\n{why[:80]}', fontsize=10, loc='left')
fig.savefig(out, dpi=110, bbox_inches='tight'); print('->', out)

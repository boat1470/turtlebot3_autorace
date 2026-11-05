#!/usr/bin/env python3
"""The tunnel on Nav2 from one run: path on the costmap, plans, and commands.

    python3 graph/tunnelnav.py ~/runs/MMDD_HHMM [out_dir]

Draws the robot's map-frame path (map->odom->base_footprint from /tf) over the
last global costmap of the tunnel, each /plan in a fading colour, and below it
the commanded v and w during tunnel_nav with collision_monitor's state.
"""

import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

TOPICS = ('/tf', '/tf_static', '/global_costmap/costmap', '/plan', '/cmd_vel',
          '/arx/nav_active', '/collision_monitor_state', '/scan')


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    x, y, t = a
    return (x + b[0] * math.cos(t) - b[1] * math.sin(t),
            y + b[0] * math.sin(t) + b[1] * math.cos(t), t + b[2])


def load(path):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = [t for t in TOPICS if t in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=want))
    cls = {t: get_message(types[t]) for t in want}
    d = {t: [] for t in want}
    map_odom = None
    while r.has_next():
        t, raw, ts = r.read_next()
        m = deserialize_message(raw, cls[t])
        s = ts * 1e-9
        if t == '/tf':
            for tr in m.transforms:
                p, q = tr.transform.translation, tr.transform.rotation
                pose = (p.x, p.y, yaw_of(q))
                if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                    map_odom = pose
                elif tr.header.frame_id == 'odom' and map_odom is not None:
                    d[t].append((s, *compose(map_odom, pose)))
        elif t == '/global_costmap/costmap':
            d[t].append((s, m))
        elif t == '/plan':
            d[t].append((s, [(p.pose.position.x, p.pose.position.y) for p in m.poses]))
        elif t == '/cmd_vel':
            tw = m.twist if hasattr(m, 'twist') else m
            d[t].append((s, tw.linear.x, tw.angular.z))
        elif t == '/arx/nav_active':
            d[t].append((s, m.data))
        elif t == '/collision_monitor_state':
            d[t].append((s, m.action_type, m.polygon_name))
    return d


def main():
    run = sys.argv[1].rstrip('/')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'debug', 'graph', os.path.basename(run))
    os.makedirs(out, exist_ok=True)
    d = load(run)
    on = [s for s, v in d.get('/arx/nav_active', []) if v]
    # nav_active drops during each back-off between goals, so the end is the
    # first drop after the last time it came on.
    off = [s for s, v in d.get('/arx/nav_active', []) if not v and on and s > on[-1]]
    t0 = on[0] if on else d['/plan'][0][0]
    t1 = off[0] if off else d['/cmd_vel'][-1][0]
    print(f'nav active {t1 - t0:.1f} s')

    fig = plt.figure(figsize=(12, 13))
    ax = fig.add_axes([0.05, 0.38, 0.9, 0.6])
    cm = [m for s, m in d['/global_costmap/costmap'] if s <= t1 + 1][-1]
    g = np.array(cm.data, dtype=float).reshape(cm.info.height, cm.info.width)
    g[g < 0] = np.nan
    r, ox, oy = cm.info.resolution, cm.info.origin.position.x, cm.info.origin.position.y
    ax.imshow(g, origin='lower', cmap='Greys', vmin=0, vmax=100,
              extent=[ox, ox + cm.info.width * r, oy, oy + cm.info.height * r])
    plans = [(s, p) for s, p in d['/plan'] if t0 - 1 <= s <= t1 + 1]
    cmap = plt.get_cmap('viridis')
    for i, (s, p) in enumerate(plans):
        if p:
            xs, ys = zip(*p)
            ax.plot(xs, ys, color=cmap(i / max(1, len(plans) - 1)), lw=0.8, alpha=0.7)
    path = [(s, x, y, th) for s, x, y, th in d['/tf'] if t0 <= s <= t1]
    ax.plot([p[1] for p in path], [p[2] for p in path], 'r', lw=2, label='robot (map)')
    for s, x, y, th in path[::15]:
        ax.annotate('', (x + 0.06 * math.cos(th), y + 0.06 * math.sin(th)), (x, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax.text(x, y, f'{s - t0:.0f}', fontsize=7, color='darkred')
    ax.plot(1.85, 1.90, 'b*', ms=14, label='goal')
    xs = [p[1] for p in path] + [0, 1.85]
    ys = [p[2] for p in path] + [0, 1.90]
    ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    ax.set_ylim(min(ys) - 0.3, max(ys) + 0.3)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_title(f'{os.path.basename(run)} tunnel: {len(plans)} plans (dark=first, yellow=last), '
                 f'numbers = s since Nav2 on')

    bx = fig.add_axes([0.07, 0.06, 0.86, 0.27])
    cv = [(s - t0, v, w) for s, v, w in d['/cmd_vel'] if t0 <= s <= t1]
    bx.plot([c[0] for c in cv], [c[2] for c in cv], 'C1', lw=1, label='w cmd (rad/s)')
    bx.plot([c[0] for c in cv], [c[1] * 10 for c in cv], 'C0', lw=1, label='v cmd x10 (m/s)')
    hd = [(s - t0, math.degrees(th)) for s, x, y, th in path]
    b2 = bx.twinx()
    b2.plot([h[0] for h in hd], [h[1] for h in hd], 'r', lw=0.8, alpha=0.6)
    b2.set_ylabel('heading on map (deg)', color='r')
    names = {0: 'none', 1: 'stop', 2: 'slow', 3: 'approach', 4: 'limit'}
    for s, a, n in d.get('/collision_monitor_state', []):
        if t0 <= s <= t1 and a:
            bx.axvline(s - t0, color={1: 'red', 2: 'orange', 3: 'purple'}.get(a, 'grey'),
                       alpha=0.15, lw=2)
    for s, p in plans:
        bx.axvline(s - t0, color='green', ls=':', lw=0.6)
    bx.axhline(0, color='k', lw=0.5)
    bx.set_xlabel('s since Nav2 on (green dotted = new plan, shading = collision_monitor '
                  'orange slow / purple approach / red stop)')
    bx.legend(loc='upper left')
    bx.grid(alpha=0.3)
    f = os.path.join(out, 'tunnelnav.png')
    fig.savefig(f, dpi=80)
    print(f)
    # Sign flips of w larger than 0.3 rad/s each way - the weaving.
    flips = sum(1 for a, b in zip(cv, cv[1:]) if a[2] * b[2] < 0 and abs(a[2] - b[2]) > 0.6)
    print(f'w sign flips (>0.3 each way): {flips}, plans {len(plans)}')
    acts = {}
    for s, a, n in d.get('/collision_monitor_state', []):
        if t0 <= s <= t1:
            acts[names.get(a, a)] = acts.get(names.get(a, a), 0) + 1
    print('collision_monitor msgs:', acts)


if __name__ == '__main__':
    main()

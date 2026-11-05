#!/usr/bin/env python3
"""The level crossing from one run: ranges, the camera's bar, and speed.

    python3 graph/levelplot.py ~/runs/MMDD_HHMM [out.png]

Top: the ultrasonic's /detect/level_range and the camera's range on its DOWN
frames, against slow_m / clear_m / stop_m read from param/mission.yaml.
Middle: /detect/level_bar. Bottom: commanded and odom speed. Grey lines are
mission_control's stage changes and stop / lift lines. From 8 s into
drive_to_level to 2 s after the crossing is done. Prints the ultrasonic
around the stop, while waiting and at the lift.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS = ('/detect/level_range', '/detect/level_range_cam', '/detect/level_bar',
          '/odom', '/cmd_vel', '/rosout')


def thresholds():
    path = os.path.join(HERE, '..', 'arx_mission', 'param', 'mission.yaml')
    level = yaml.safe_load(open(path))['/**']['ros__parameters']['level']
    return level['slow_m'], level['clear_m'], level['stop_m']


def load(run):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=run, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = [t for t in TOPICS if t in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=want))
    cls = {t: get_message(types[t]) for t in want}
    d = {t: [] for t in TOPICS}
    events = []
    while r.has_next():
        t, raw, ts = r.read_next()
        s = ts * 1e-9
        m = deserialize_message(raw, cls[t])
        if t == '/rosout':
            if m.name == 'arx_mission_control' and any(
                    k in m.msg for k in ('stage', 'stopping', 'lifted', 'crossing bar', 'crept')):
                events.append((s, m.msg))
        elif t == '/odom':
            d[t].append((s, m.twist.twist.linear.x))
        elif t == '/cmd_vel':
            tw = m.twist if hasattr(m, 'twist') else m
            d[t].append((s, tw.linear.x))
        else:
            d[t].append((s, m.data))
    return d, events


def main():
    run = sys.argv[1].rstrip('/')
    name = os.path.basename(run)
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        HERE, '..', 'debug', 'graph', name, 'level.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d, ev = load(run)
    starts = [s for s, m in ev if '-> drive_to_level' in m]
    if not starts:
        print('no drive_to_level in this run')
        return
    t0 = starts[0]
    ends = [s for s, m in ev if s > t0 and ('-> drive_to_tunnel' in m or '-> follow_side' in m)]
    t1 = ends[0] if ends else max(s for s, _ in d['/odom'])
    a, b = t0 + 8, t1 + 2

    def seg(k, keep=lambda v: True):
        return [(s - t0, v) for s, v in d[k] if a <= s <= b and keep(v)]

    slow_m, clear_m, stop_m = thresholds()
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    us = seg('/detect/level_range')
    cam = seg('/detect/level_range_cam', lambda v: v < 5)
    ax[0].plot([s for s, _ in us], [v for _, v in us], 'b.-', ms=3, lw=0.8,
               label='ultrasonic /detect/level_range')
    ax[0].plot([s for s, _ in cam], [v for _, v in cam], 'r.', ms=4,
               label='camera range (DOWN frames)')
    for y, c, lab in ((slow_m, 'orange', f'slow_m {slow_m}'), (clear_m, 'green', f'clear_m {clear_m}'),
                      (stop_m, 'red', f'stop_m {stop_m}')):
        ax[0].axhline(y, color=c, ls='--', lw=1, label=lab)
    ax[0].set_ylim(0, 1.2)
    ax[0].set_ylabel('m')
    ax[0].legend(fontsize=8, loc='upper right')
    ax[0].grid(alpha=.3)
    ax[0].set_title(f'{name} level crossing')
    bar = seg('/detect/level_bar')
    ax[1].plot([s for s, _ in bar], [v for _, v in bar], 'k.', ms=3)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(['nothing', 'UP', 'DOWN'])
    ax[1].set_ylabel('camera bar')
    ax[1].grid(alpha=.3)
    o, c = seg('/odom'), seg('/cmd_vel')
    ax[2].plot([s for s, _ in c], [v for _, v in c], 'C1', label='cmd v')
    ax[2].plot([s for s, _ in o], [v for _, v in o], 'C0', label='odom v')
    ax[2].set_ylabel('m/s')
    ax[2].set_xlabel('s since drive_to_level')
    ax[2].legend()
    ax[2].grid(alpha=.3)
    for s, m in ev:
        if a <= s <= b:
            for x in ax:
                x.axvline(s - t0, color='grey', lw=0.8)
            ax[0].text(s - t0, 1.15, m.split(':')[0][:28] if m.startswith('stage') else m[:22],
                       rotation=90, fontsize=7, va='top')
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    print(out)

    stops = [s for s, m in ev if 'stopping' in m or 'crept' in m]
    rng = d['/detect/level_range']
    if not rng:
        print('no ultrasonic range in this run')
        return
    print(f'ultrasonic {len([1 for s, _ in rng if a <= s <= b]) / (b - a):.1f} Hz')
    if stops:
        st = stops[0]
        print('around the stop:', [round(v, 3) for s, v in rng if st - 1 <= s <= st + 0.5])
        wait = [v for s, v in rng if st + 0.5 <= s <= t1 - 0.6]
        if wait:
            print(f'waiting: n {len(wait)} min {min(wait):.3f} max {max(wait):.3f} '
                  f'median {sorted(wait)[len(wait) // 2]:.3f}')
    print('around the lift:', [round(v, 2) for s, v in rng if t1 - 1.2 <= s <= t1 + 0.3])


if __name__ == '__main__':
    main()

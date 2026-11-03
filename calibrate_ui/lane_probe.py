#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Look at the lane lines the way detect_lane's thresholds see them, live.

    python3 calibrate_ui/lane_probe.py                          /camera/image_projected
    python3 calibrate_ui/lane_probe.py /detect/image_input/compressed

roi_probe.py measures a box for the camera's own settings; this one measures
it against the lane thresholds. The window shows /camera/image_projected - the
raw Image detect_lane steers by - with what the white and yellow thresholds
in arx_mission/param/lane.yaml let through painted over it, and the two rows
detect_lane reads drawn across: lookahead_row (solid) and speed_row (dashed).

Drag inside the box to move it, its corner to resize it, anywhere else to
draw a new one. Inside the box, ten times a second:

    H S V        1st-99th and 5th-95th percentile, OpenCV units (H 0-179)
    white/yellow how much of the box each threshold passes
    headroom     for the pixels that pass, how far the dimmest 5% sit above
                 the floor that keeps them in (V for white, S and V for
                 yellow). A few units means a cloud or a shadow drops them
    line x       the middle of the passing pixels, and how much it moves
                 frame to frame (std of the second difference, px) - the
                 same jitter measure the run analysis uses on /detect/lane
    count        how much the number of passing pixels moves frame to frame

So: put the box on a stretch of white line near lookahead_row and watch line
x jitter and headroom while the light changes, or the robot creeps along.
A torn or dirty bit of tape shows as a hole in the overlay and a jump in
count as it passes through the box.

Freeze holds one frame to look at. Reload re-reads lane.yaml - values tuned
in rqt are not seen until they are saved there. detect_lane also gates white
pixels left of the yellow line (lane.white_gate); that is not applied here.
Read-only: subscribes and draws, nothing else.
"""

import argparse
import collections
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import yaml

from lane_width import to_bgr

LANE_YAML = Path(__file__).resolve().parent.parent / 'arx_mission' / 'param' / 'lane.yaml'
VIEW = (800, 480)       # the image is scaled to fit this
HISTORY = 30            # frames for the jitter numbers, ~1 s
WHITE_TINT = (255, 0, 255)     # BGR, magenta: shows on both mat and line
YELLOW_TINT = (255, 255, 0)    # cyan


def load_lane_yaml():
    p = yaml.safe_load(LANE_YAML.read_text())['/**']['ros__parameters']

    def band(c):
        return (np.array([c['hue_l'], c['saturation_l'], c['lightness_l']]),
                np.array([c['hue_h'], c['saturation_h'], c['lightness_h']]))
    lane = p['lane']
    return {'white': band(p['detect']['lane']['white']),
            'yellow': band(p['detect']['lane']['yellow']),
            'lookahead_row': lane['lookahead_row'],
            'speed_row': lane.get('speed_row', 0)}


def jitter(xs):
    x = np.asarray([v for v in xs if v is not None], float)
    if x.size < 5:
        return None
    return float(np.std(x[1:-1] - (x[:-2] + x[2:]) / 2))


class Probe(Node):

    def __init__(self, topic):
        super().__init__('lane_probe')
        self.lock = threading.Lock()
        self.frame, self.stamp = None, 0.0
        self.stamps = collections.deque(maxlen=60)
        kind = CompressedImage if topic.endswith('/compressed') else Image
        self.create_subscription(kind, topic, self.on_image, 1)

    def on_image(self, msg):
        img = to_bgr(msg)
        if img is None:
            return
        now = time.monotonic()
        with self.lock:
            self.frame, self.stamp = img, now
            self.stamps.append(now)

    def latest(self):
        with self.lock:
            hz = 0.0
            if len(self.stamps) > 1:
                hz = (len(self.stamps) - 1) / max(self.stamps[-1] - self.stamps[0], 1e-6)
            return self.frame, self.stamp, hz


class Window:

    def __init__(self, node, topic, size):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image as PilImage
        from PIL import ImageTk
        self.tk, self.pil, self.pil_tk = tk, PilImage, ImageTk
        self.node, self.topic, self.size = node, topic, size
        self.cfg = load_lane_yaml()
        self.roi = None
        self.img_wh = None
        self.scale = 1.0
        self.drag = None
        self.photo = None
        self.shown = 0.0
        self.frozen = None
        self.hist = {k: collections.deque(maxlen=HISTORY)
                     for k in ('wx', 'yx', 'wn', 'yn')}

        root = self.root = tk.Tk()
        root.title('arx lane probe')
        mono = ('TkFixedFont', 10)

        self.canvas = tk.Canvas(root, width=VIEW[0], height=VIEW[1], bg='black',
                                highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, padx=8, pady=8, sticky='n')
        self.img_item = self.canvas.create_image(0, 0, anchor='nw')
        self.look_line = self.canvas.create_line(0, 0, 0, 0, fill='white', width=1)
        self.speed_line = self.canvas.create_line(0, 0, 0, 0, fill='#aaa', dash=(4, 4))
        self.box = self.canvas.create_rectangle(0, 0, 0, 0, outline='#00ff66', width=2)
        self.handle = self.canvas.create_rectangle(0, 0, 0, 0, fill='#00ff66', outline='')
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'drag', None))

        side = ttk.Frame(root, padding=(0, 8, 8, 8))
        side.grid(row=0, column=1, sticky='n')

        show = ttk.LabelFrame(side, text='overlay', padding=6)
        show.pack(fill='x')
        self.overlay = tk.StringVar(value='both')
        for v, t in (('none', 'image only'), ('white', 'white mask'),
                     ('yellow', 'yellow mask'), ('both', 'both')):
            ttk.Radiobutton(show, text=t, value=v, variable=self.overlay,
                            command=self.redraw).pack(side='left')

        self.stats = tk.StringVar(value=f'waiting for {topic} ...')
        tk.Label(side, textvariable=self.stats, font=mono, justify='left', anchor='w',
                 width=50, bg='#111', fg='#ddd', padx=8, pady=6).pack(fill='x', pady=(8, 0))

        self.bands = tk.StringVar()
        ttk.Label(side, textvariable=self.bands, font=mono, foreground='#555',
                  justify='left').pack(fill='x', pady=(6, 0))

        bar = ttk.Frame(side)
        bar.pack(fill='x', pady=(8, 0))
        self.freeze_btn = ttk.Button(bar, text='Freeze', command=self.toggle_freeze)
        self.freeze_btn.pack(side='left')
        ttk.Button(bar, text='Reload lane.yaml', command=self.reload).pack(side='left', padx=6)
        ttk.Button(bar, text='Box on lookahead row', command=self.box_on_row).pack(side='left')

        self.status = tk.StringVar(value='')
        ttk.Label(root, textvariable=self.status, foreground='#555',
                  padding=(8, 0, 8, 8)).grid(row=1, column=0, columnspan=2, sticky='w')
        self.show_bands()
        root.after(100, self.tick)

    # --- the box -------------------------------------------------------

    def clamp(self):
        w_img, h_img = self.img_wh
        x, y, w, h = self.roi
        w, h = max(2, min(w, w_img)), max(2, min(h, h_img))
        self.roi = [max(0, min(x, w_img - w)), max(0, min(y, h_img - h)), w, h]

    def press(self, e):
        if self.roi is None:
            return
        x, y, w, h = self.roi
        s = self.scale
        ix, iy = e.x / s, e.y / s
        if abs(e.x - (x + w) * s) < 10 and abs(e.y - (y + h) * s) < 10:
            self.drag = ('size', x, y)
        elif x <= ix < x + w and y <= iy < y + h:
            self.drag = ('move', ix - x, iy - y)
        else:
            self.drag = ('new', ix, iy)
            self.roi = [int(ix), int(iy), 2, 2]
        self.motion(e)

    def motion(self, e):
        if not self.drag:
            return
        ix, iy = e.x / self.scale, e.y / self.scale
        kind, a, b = self.drag
        if kind == 'move':
            self.roi[0], self.roi[1] = int(ix - a), int(iy - b)
        else:
            x0, x1 = sorted((a, ix))
            y0, y1 = sorted((b, iy))
            self.roi = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
        self.clamp()
        self.moved_box()

    def moved_box(self):
        for d in self.hist.values():
            d.clear()
        self.draw_box()
        self.redraw()

    def box_on_row(self):
        if self.roi is None:
            return
        row = self.cfg['lookahead_row']
        self.roi[1] = row - self.roi[3] // 2
        self.clamp()
        self.moved_box()

    def draw_box(self):
        x, y, w, h = self.roi
        s = self.scale
        self.canvas.coords(self.box, x * s, y * s, (x + w) * s, (y + h) * s)
        self.canvas.coords(self.handle, (x + w) * s - 5, (y + h) * s - 5,
                           (x + w) * s + 5, (y + h) * s + 5)

    # --- buttons -------------------------------------------------------

    def toggle_freeze(self):
        if self.frozen is None:
            frame, _, _ = self.node.latest()
            if frame is None:
                return
            self.frozen = frame.copy()
            self.freeze_btn.config(text='Live')
            self.status.set('frozen - box numbers are for this one frame')
        else:
            self.frozen = None
            self.freeze_btn.config(text='Freeze')
            self.status.set('')
        self.redraw()

    def reload(self):
        try:
            self.cfg = load_lane_yaml()
        except Exception as ex:     # a half-saved file should not kill the window
            self.status.set(f'lane.yaml: {ex}')
            return
        self.show_bands()
        for d in self.hist.values():
            d.clear()
        self.status.set('lane.yaml reloaded')
        self.redraw()

    def show_bands(self):
        c = self.cfg

        def fmt(name):
            lo, hi = c[name]
            return (f'{name:6s} H {lo[0]}-{hi[0]}  S {lo[1]}-{hi[1]}  V {lo[2]}-{hi[2]}')
        self.bands.set(f'lane.yaml\n{fmt("white")}\n{fmt("yellow")}\n'
                       f'lookahead_row {c["lookahead_row"]}  speed_row {c["speed_row"]}')

    # --- the loop ------------------------------------------------------

    def tick(self):
        frame, stamp, hz = self.node.latest()
        if self.frozen is None and frame is not None and stamp != self.shown:
            self.shown = stamp
            self.hz = hz
            self.render(frame, live=True)
        self.root.after(100, self.tick)

    def redraw(self):
        frame = self.frozen if self.frozen is not None else self.node.latest()[0]
        if frame is not None:
            self.render(frame, live=False)

    def render(self, frame, live):
        h, w = frame.shape[:2]
        if self.img_wh != (w, h):
            self.img_wh = (w, h)
            self.scale = min(VIEW[0] / w, VIEW[1] / h)
            self.canvas.config(width=int(w * self.scale), height=int(h * self.scale))
            bw, bh = min(self.size, w), min(self.size // 2, h)
            # Start on the white line's side at lookahead_row: where the
            # steering is read, and where a white line usually is.
            self.roi = [int(w * 0.62), self.cfg['lookahead_row'] - bh // 2, bw, bh]
            self.clamp()
            self.draw_box()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = {k: cv2.inRange(hsv, *self.cfg[k]) for k in ('white', 'yellow')}

        view = frame.copy()
        mode = self.overlay.get()
        for k, tint in (('white', WHITE_TINT), ('yellow', YELLOW_TINT)):
            if mode in (k, 'both'):
                view[masks[k] > 0] = tint
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        img = self.pil.fromarray(rgb).resize(
            (int(w * self.scale), int(h * self.scale)), self.pil.BILINEAR)
        self.photo = self.pil_tk.PhotoImage(img)
        self.canvas.itemconfig(self.img_item, image=self.photo)
        s = self.scale
        for item, row in ((self.look_line, self.cfg['lookahead_row']),
                          (self.speed_line, self.cfg['speed_row'])):
            y = row * s if 0 < row < h else -10
            self.canvas.coords(item, 0, y, w * s, y)
        self.canvas.tag_raise(self.box)
        self.canvas.tag_raise(self.handle)

        x, y, bw, bh = self.roi
        self.stats.set(self.measure(hsv[y:y + bh, x:x + bw],
                                    {k: m[y:y + bh, x:x + bw] for k, m in masks.items()},
                                    x, live))

    def measure(self, hsv, masks, x0, live):
        px = hsv.reshape(-1, 3)
        p1, p5, p95, p99 = np.percentile(px, (1, 5, 95, 99), axis=0)
        n = px.shape[0]
        lines = [f'{self.img_wh[0]}x{self.img_wh[1]}  {getattr(self, "hz", 0):4.1f} Hz   '
                 f'box {hsv.shape[1]}x{hsv.shape[0]} at ({self.roi[0]}, {self.roi[1]})',
                 f'H {p5[0]:3.0f}-{p95[0]:3.0f}  S {p5[1]:3.0f}-{p95[1]:3.0f}  '
                 f'V {p5[2]:3.0f}-{p95[2]:3.0f}   (p5-p95)',
                 f'H {p1[0]:3.0f}-{p99[0]:3.0f}  S {p1[1]:3.0f}-{p99[1]:3.0f}  '
                 f'V {p1[2]:3.0f}-{p99[2]:3.0f}   (p1-p99)', '']
        for k, xkey, nkey in (('white', 'wx', 'wn'), ('yellow', 'yx', 'yn')):
            m = masks[k] > 0
            cnt = int(m.sum())
            lo, hi = self.cfg[k]
            lines.append(f'{k:6s} passes {cnt:6d} px  {100 * cnt / max(n, 1):5.1f}% of box')
            if cnt:
                on = hsv[m]
                v5 = np.percentile(on[:, 2], 5)
                head = f'  headroom V {v5 - lo[2]:+4.0f}'
                if k == 'yellow':
                    head += f'  S {np.percentile(on[:, 1], 5) - lo[1]:+4.0f}'
                else:
                    # white is kept in by a ceiling on S, not a floor
                    head += f'  S under {hi[1] - np.percentile(on[:, 1], 95):+4.0f}'
                lines.append(head + '  (dimmest 5%)')
                cols = np.flatnonzero(m.any(axis=0))
                lx = x0 + float((np.arange(m.shape[1]) * m.sum(axis=0)).sum() / cnt)
                lines.append(f'  line x {lx:6.1f}  across {cols.size} px')
            else:
                lx = None
                lines.append('  none in the box')
            if live:
                self.hist[xkey].append(lx)
                self.hist[nkey].append(cnt)
            jx = jitter(self.hist[xkey])
            cn = np.asarray(self.hist[nkey], float)
            cj = (f'{100 * cn.std() / cn.mean():4.1f}%' if cn.size >= 5 and cn.mean() > 0
                  else '  - ')
            lines.append(f'  jitter: line x {jx:4.2f} px   count {cj}' if jx is not None
                         else f'  jitter: line x  -     count {cj}')
            lines.append('')
        lines.append(f'jitter over the last {len(self.hist["wn"])} frames '
                     f'(of {HISTORY}); moving the box restarts it')
        return '\n'.join(lines)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('topic', nargs='?', default='/camera/image_projected')
    ap.add_argument('--size', type=int, default=80, help='starting box width, px')
    args = ap.parse_args()

    rclpy.init()
    node = Probe(args.topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        Window(node, args.topic, args.size).run()
    except KeyboardInterrupt:
        pass
    finally:
        # As roi_probe: stop the spinning thread before the node goes.
        executor.shutdown(timeout_sec=1.0)
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

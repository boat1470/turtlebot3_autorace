#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Watch what detect_lane sees and reports, in one window.

    python3 calibrate_ui/lane_debug.py                  image + numbers
    python3 calibrate_ui/lane_debug.py --no-image       numbers only, light on Wi-Fi

Everything detect_lane publishes for debugging, without four `ros2 topic
echo` terminals and an rqt_image_view:

    image        /detect/image_lane, or either mask - pick with the buttons.
                 The masks exist only with lane.publish_debug_image: true,
                 which detect_lane reads at startup, so set it in lane.yaml
                 and restart the node. Only the image shown is subscribed.
    lane state   /detect/lane_state as words: none / yellow only / both /
                 white only
    error        /detect/lane - 500, the number control_lane steers by, on a
                 bar from -500 to +500, with the speed factor
                 (1 - |error|/500)^2.2 that the same error sets. Negative
                 means the centre is left of the robot
    history      the last 10 s of error. Gaps are frames with no centre:
                 detect_lane publishes nothing on /detect/lane then
    reliability  /detect/white_line_reliability and yellow, 0-100
    Hz           how often /detect/lane arrives

Topic names are the ones detect_lane.launch.py remaps to. A value turns grey
with "stale" when nothing has arrived for a second.
"""

import argparse
import collections
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from std_msgs.msg import UInt8

IMAGES = (
    ('lane', '/detect/image_lane/compressed'),
    ('white mask', '/detect/image_white_lane_marker/compressed'),
    ('yellow mask', '/detect/image_yellow_lane_marker/compressed'),
)

# lane_state -> text, colour
STATES = {
    0: ('0  NO LANE', '#e04040'),
    1: ('1  YELLOW ONLY (left)', '#e0b000'),
    2: ('2  BOTH LINES', '#30b050'),
    3: ('3  WHITE ONLY (right)', '#d0d0d0'),
}

CENTER = 500        # control_lane: error = center - 500
STALE = 1.0         # s with no message before a value is shown as stale
HISTORY = 10.0      # s of error kept for the plot
VIEW = (500, 300)   # the image is scaled to fit this
BAR_W, BAR_H = 420, 28
PLOT_H = 110


class Listener(Node):
    """Keeps the latest of everything. Spun in its own thread."""

    def __init__(self, with_image):
        super().__init__('lane_debug')
        self.lock = threading.Lock()
        self.values = {}                     # topic -> (value, time)
        self.errors = collections.deque()    # (time, error)
        self.lane_times = collections.deque()
        self.frame, self.frame_stamp = None, 0.0
        self.image_sub = None

        for topic, kind in (('/detect/lane', Float64),
                            ('/detect/lane_state', UInt8),
                            ('/detect/white_line_reliability', UInt8),
                            ('/detect/yellow_line_reliability', UInt8)):
            self.create_subscription(
                kind, topic, lambda msg, t=topic: self.on_value(t, msg.data), 10)
        if with_image:
            self.watch(IMAGES[0][1])

    def on_value(self, topic, value):
        now = time.monotonic()
        with self.lock:
            self.values[topic] = (value, now)
            if topic == '/detect/lane':
                self.errors.append((now, value - CENTER))
                self.lane_times.append(now)
                while self.errors and now - self.errors[0][0] > HISTORY:
                    self.errors.popleft()
                while self.lane_times and now - self.lane_times[0] > 2.0:
                    self.lane_times.popleft()

    def on_image(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with self.lock:
                self.frame, self.frame_stamp = img, time.monotonic()

    def watch(self, topic):
        """Subscribe to one image topic only, so the others cost no Wi-Fi."""
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
        with self.lock:
            self.frame, self.frame_stamp = None, 0.0
        self.image_sub = self.create_subscription(CompressedImage, topic, self.on_image, 1)

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            lt = self.lane_times
            hz = (len(lt) - 1) / (lt[-1] - lt[0]) if len(lt) > 1 and lt[-1] > lt[0] else 0.0
            if lt and now - lt[-1] > STALE:
                hz = 0.0
            return (dict(self.values), list(self.errors), hz,
                    self.frame, self.frame_stamp)


class Window:

    def __init__(self, node, with_image):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image as PilImage
        from PIL import ImageTk
        self.tk, self.pil, self.pil_tk = tk, PilImage, ImageTk
        self.node, self.with_image = node, with_image
        self.photo, self.shown = None, 0.0

        root = self.root = tk.Tk()
        root.title('arx lane debug')
        root.configure(bg='#1b1b1b')
        style = ttk.Style(root)
        style.configure('Dark.TFrame', background='#1b1b1b')
        big = ('TkDefaultFont', 18, 'bold')
        mono = ('TkFixedFont', 11)
        fg, dim = '#e8e8e8', '#777'

        def label(parent, **kw):
            kw.setdefault('bg', '#1b1b1b')
            kw.setdefault('fg', fg)
            return tk.Label(parent, **kw)

        col = 0
        if with_image:
            left = ttk.Frame(root, style='Dark.TFrame', padding=8)
            left.grid(row=0, column=0, sticky='n')
            self.canvas = tk.Canvas(left, width=VIEW[0], height=VIEW[1],
                                    bg='black', highlightthickness=0)
            self.canvas.pack()
            self.img_item = self.canvas.create_image(0, 0, anchor='nw')
            self.no_img = self.canvas.create_text(
                VIEW[0] // 2, VIEW[1] // 2, fill=dim, text='waiting for an image...')
            pick = ttk.Frame(left, style='Dark.TFrame')
            pick.pack(fill='x', pady=(6, 0))
            self.which = tk.StringVar(value=IMAGES[0][1])
            for name, topic in IMAGES:
                tk.Radiobutton(pick, text=name, value=topic, variable=self.which,
                               command=self.switch, indicatoron=False, width=12,
                               bg='#2a2a2a', fg=fg, selectcolor='#3a5a8a',
                               activebackground='#3a3a3a', activeforeground=fg
                               ).pack(side='left', padx=(0, 4))
            self.img_info = tk.StringVar(value='')
            label(left, textvariable=self.img_info, fg=dim, anchor='w',
                  justify='left').pack(fill='x', pady=(4, 0))
            col = 1

        right = ttk.Frame(root, style='Dark.TFrame', padding=8)
        right.grid(row=0, column=col, sticky='n')

        label(right, text='lane state', fg=dim, anchor='w').pack(fill='x')
        self.state = label(right, text='-', font=big, anchor='w')
        self.state.pack(fill='x')

        label(right, text='error  (/detect/lane - 500)', fg=dim, anchor='w'
              ).pack(fill='x', pady=(10, 0))
        self.err_text = label(right, text='-', font=mono, anchor='w')
        self.err_text.pack(fill='x')
        self.bar = tk.Canvas(right, width=BAR_W, height=BAR_H, bg='#111',
                             highlightthickness=0)
        self.bar.pack(anchor='w', pady=(2, 0))
        mid = BAR_W / 2
        self.bar.create_line(mid, 0, mid, BAR_H, fill='#555')
        for e in (-250, 250):
            x = mid + e / CENTER * mid
            self.bar.create_line(x, BAR_H - 6, x, BAR_H, fill='#444')
        self.bar_fill = self.bar.create_rectangle(mid, 4, mid, BAR_H - 4, outline='')
        self.bar_mark = self.bar.create_line(mid, 0, mid, BAR_H, width=3, fill=fg)
        ticks = ttk.Frame(right, style='Dark.TFrame')
        ticks.pack(fill='x')
        for text, side in (('-500  left', 'left'), ('right  +500', 'right')):
            label(ticks, text=text, fg=dim, font=('TkDefaultFont', 8)).pack(side=side)

        label(right, text=f'error, last {HISTORY:.0f} s', fg=dim, anchor='w'
              ).pack(fill='x', pady=(10, 0))
        self.plot = tk.Canvas(right, width=BAR_W, height=PLOT_H, bg='#111',
                              highlightthickness=0)
        self.plot.pack(anchor='w')

        label(right, text='line reliability', fg=dim, anchor='w'
              ).pack(fill='x', pady=(10, 0))
        self.rel = {}
        for name, topic, colour in (
                ('yellow', '/detect/yellow_line_reliability', '#e0b000'),
                ('white', '/detect/white_line_reliability', '#d0d0d0')):
            row = ttk.Frame(right, style='Dark.TFrame')
            row.pack(fill='x', pady=2)
            label(row, text=name, width=7, anchor='w').pack(side='left')
            c = tk.Canvas(row, width=300, height=18, bg='#111', highlightthickness=0)
            c.pack(side='left')
            fill = c.create_rectangle(0, 0, 0, 18, fill=colour, outline='')
            text = label(row, text='-', font=mono, width=10, anchor='w')
            text.pack(side='left', padx=(6, 0))
            self.rel[topic] = (c, fill, text)

        self.footer = tk.StringVar(value='')
        label(right, textvariable=self.footer, font=mono, fg=dim, anchor='w',
              justify='left').pack(fill='x', pady=(10, 0))

        root.after(100, self.tick)

    def switch(self):
        if rclpy.ok():
            self.node.watch(self.which.get())
            self.shown = 0.0

    @staticmethod
    def err_colour(e):
        a = abs(e)
        return '#30b050' if a < 50 else '#e0b000' if a < 150 else '#e04040'

    def tick(self):
        # rclpy takes Ctrl+C and SIGTERM for itself and shuts the context
        # down, which Tk never hears about: close the window here instead.
        if not rclpy.ok():
            self.root.destroy()
            return
        now = time.monotonic()
        values, errors, hz, frame, stamp = self.node.snapshot()

        def get(topic):
            v = values.get(topic)
            if v is None:
                return None, True
            return v[0], now - v[1] > STALE

        # lane state
        s, stale = get('/detect/lane_state')
        if s is None:
            self.state.config(text='no data - is detect_lane running?', fg='#777')
        else:
            text, colour = STATES.get(s, (str(s), '#e8e8e8'))
            self.state.config(text=text + ('   (stale)' if stale else ''),
                              fg='#777' if stale else colour)

        # error
        c, stale = get('/detect/lane')
        mid = BAR_W / 2
        if c is None:
            self.err_text.config(text='no /detect/lane yet', fg='#777')
            self.bar.coords(self.bar_fill, mid, 4, mid, BAR_H - 4)
            self.bar.coords(self.bar_mark, mid, 0, mid, BAR_H)
        else:
            e = c - CENTER
            speed = max(1 - abs(e) / CENTER, 0) ** 2.2
            colour = '#777' if stale else self.err_colour(e)
            self.err_text.config(
                text=f'{e:+7.1f} px   centre {c:6.1f}   speed x{speed:.2f}'
                     + ('   (stale)' if stale else ''), fg=colour)
            x = mid + max(-1.0, min(1.0, e / CENTER)) * mid
            self.bar.coords(self.bar_fill, min(mid, x), 4, max(mid, x), BAR_H - 4)
            self.bar.itemconfig(self.bar_fill, fill=colour)
            self.bar.coords(self.bar_mark, x, 0, x, BAR_H)
        self.draw_plot(errors, now)

        # reliability
        for topic, (canvas, fill, text) in self.rel.items():
            r, stale = get(topic)
            if r is None:
                canvas.coords(fill, 0, 0, 0, 18)
                text.config(text='-', fg='#777')
            else:
                canvas.coords(fill, 0, 0, 300 * min(r, 100) / 100, 18)
                warn = '#e04040' if r < 50 else '#e8e8e8'
                text.config(text=f'{r:3d}' + (' stale' if stale else ''),
                            fg='#777' if stale else warn)

        self.footer.set(f'/detect/lane  {hz:4.1f} Hz')

        if self.with_image:
            if frame is not None and stamp != self.shown:
                self.shown = stamp
                self.show(frame)
            age = now - stamp if stamp else None
            self.img_info.set(self.which.get() + (
                '' if age is None else f'   {age:.1f} s ago' if age > STALE else ''))
            if frame is None:
                self.canvas.itemconfig(self.img_item, image='')
                self.canvas.itemconfig(
                    self.no_img, state='normal',
                    text='waiting for an image...' if self.which.get() == IMAGES[0][1]
                    else 'no mask - lane.publish_debug_image: true\n'
                         'in lane.yaml, then restart detect_lane')

        self.root.after(100, self.tick)

    def draw_plot(self, errors, now):
        p = self.plot
        p.delete('all')
        h2 = PLOT_H / 2
        p.create_line(0, h2, BAR_W, h2, fill='#444')
        for e in (-250, 250):
            y = h2 - e / CENTER * h2
            p.create_line(0, y, BAR_W, y, fill='#2a2a2a', dash=(2, 4))
        seg = []
        last_t = None
        for t, e in errors:
            x = BAR_W - (now - t) / HISTORY * BAR_W
            y = h2 - max(-1.0, min(1.0, e / CENTER)) * h2
            if last_t is not None and t - last_t > 0.4:   # no centre in between
                if len(seg) >= 4:
                    p.create_line(*seg, fill='#4aa0ff', width=2)
                seg = []
            seg += [x, y]
            last_t = t
        if len(seg) >= 4:
            p.create_line(*seg, fill='#4aa0ff', width=2)

    def show(self, frame):
        h, w = frame.shape[:2]
        scale = min(VIEW[0] / w, VIEW[1] / h)
        size = (int(w * scale), int(h * scale))
        self.canvas.config(width=size[0], height=size[1])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = self.pil_tk.PhotoImage(self.pil.fromarray(rgb).resize(size))
        self.canvas.itemconfig(self.img_item, image=self.photo)
        self.canvas.itemconfig(self.no_img, state='hidden')

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--no-image', action='store_true',
                    help='numbers only, subscribe to no image')
    args = ap.parse_args()

    rclpy.init()
    node = Listener(with_image=not args.no_image)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    def spin():
        try:
            executor.spin()
        except Exception:   # the context going away under Ctrl+C or SIGTERM
            pass

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        Window(node, with_image=not args.no_image).run()
    except KeyboardInterrupt:
        pass
    finally:
        # As in roi_probe.py: stop and join the spinner before the node goes.
        executor.shutdown(timeout_sec=1.0)
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

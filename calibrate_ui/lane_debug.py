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
                 bar from -500 to +500, and beside it the braking row's error
                 (/detect/lane_ahead - 500). Negative means the centre is
                 left of the robot
    speed        what the robot was actually told, off /cmd_vel - linear and
                 angular, after control_lane's slow-down, gain scaling and
                 wheel clamp - with the speed odom measured beside it, and a
                 bar against the Burger's 0.22 m/s
    history      the last 10 s of error. Gaps are frames with no centre:
                 detect_lane publishes nothing on /detect/lane then
    reliability  /detect/white_line_reliability and yellow, 0-100
    heading      the robot's yaw off /odom, on a dial: 0 at the bottom is
                 the way it faced when robot.launch.py came up, 180 at the
                 top, + is left (counter-clockwise, so 90 is on the right) -
                 the same angle mission_control uses, drawn the way round
                 docs/curve_map.png is, where the robot sets off downwards.
                 The dashed line is --target (default 180, the left script's
                 heading), with how far is left to turn to it
    Hz           how often /detect/lane arrives, and beside it the camera
                 stamp -> detect_lane latency in ms (needs the Pi's clock and
                 this one's in step; "clocks?" flags a value that cannot be)
    sign         the 'sign' image button shows detect_sign's debug image (the
                 frame, with the reference sign and the matched points beside
                 it on a hit); above lane state, which mission is armed and
                 the last sign reported, with its age
    obstacle     the 'obstacle' image button shows detect_obstacle's top-down
                 LiDAR picture; under the sign line, which boxes ahead are
                 blocked, what is beside the robot and the nearest thing in
                 front - the robot's own left and right
    batt         a small button: reads /battery_state once and shows voltage
                 and percent as of that press. It does not update by itself
    REC          records a run: `ros2 bag record` of RECORD_TOPICS into
                 ~/runs/<month-day_hour-minute>/, red with the time while it
                 runs, press again (or close the window) to stop. The box
                 beside it adds a line to ~/runs/notes.txt about the last run,
                 with the folder name and how far into it the note was taken

Topic names are the ones detect_lane.launch.py remaps to. A value turns grey
with "stale" when nothing has arrived for a second.
"""

import argparse
import collections
import math
import os
import signal
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from std_msgs.msg import UInt8

IMAGES = (
    ('lane', '/detect/image_lane/compressed'),
    ('white mask', '/detect/image_white_lane_marker/compressed'),
    ('yellow mask', '/detect/image_yellow_lane_marker/compressed'),
    ('sign', '/detect/image_traffic_sign/compressed'),
    ('obstacle', '/detect/image_obstacle/compressed'),
)
SIGN_IMAGE = IMAGES[3][1]
OBSTACLE_IMAGE = IMAGES[4][1]

# /detect/obstacle is a bitmask, /detect/obstacle_side a value - both the
# robot's own left and right (detect_obstacle.py)
BLOCKED_BITS = ((1, 'AHEAD'), (2, 'LEFT'), (4, 'RIGHT'))
SIDES = {0: 'clear', 1: 'LEFT', 2: 'RIGHT'}

# /detect/traffic_sign and /arx/armed_mission, as mission_control numbers them
SIGNS = {1: 'CONSTRUCTION', 2: 'LEFT', 3: 'RIGHT', 4: 'PARKING'}
MISSIONS = {0: 'none', 1: 'traffic light', 2: 'intersection', 3: 'construction',
            4: 'parking', 5: 'level crossing', 6: 'tunnel'}

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
WHEEL_MAX = 0.22    # m/s, the Burger's wheel limit - the speed bar's full scale
DIAL = 130          # px, the heading dial


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
                            ('/detect/lane_ahead', Float64),
                            ('/detect/lane_state', UInt8),
                            ('/detect/white_line_reliability', UInt8),
                            ('/detect/yellow_line_reliability', UInt8),
                            ('/detect/traffic_sign', UInt8),
                            ('/arx/armed_mission', UInt8),
                            ('/detect/obstacle', UInt8),
                            ('/detect/obstacle_side', UInt8),
                            ('/detect/front_range', Float32),
                            ('/detect/lane_latency_ms', Float64)):
            self.create_subscription(
                kind, topic, lambda msg, t=topic: self.on_value(t, msg.data), 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        # control_lane's output, remapped to /cmd_vel; TwistStamped, the type
        # the Jazzy turtlebot3_node takes.
        self.create_subscription(
            TwistStamped, '/cmd_vel',
            lambda m: self.on_value('/cmd_vel', (m.twist.linear.x, m.twist.angular.z)), 10)
        self.create_subscription(
            BatteryState, '/battery_state',
            lambda m: self.on_value('/battery_state', (m.voltage, m.percentage)), 1)
        if with_image:
            self.watch(IMAGES[0][1])

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.degrees(math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        self.on_value('/odom', yaw)
        self.on_value('/odom_v', msg.twist.twist.linear.x)

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


RUNS = os.path.expanduser('~/runs')
# What a run is recorded with - see notes/CLAUDE.md, "Recording a real-robot
# run". /rosout carries every node's log, stage changes included.
RECORD_TOPICS = (
    '/detect/image_lane/compressed', '/detect/image_yellow_lane_marker/compressed',
    '/detect/image_white_lane_marker/compressed',
    '/detect/lane', '/detect/lane_ahead', '/detect/lane_state',
    '/detect/yellow_line_reliability', '/detect/white_line_reliability',
    '/detect/traffic_sign', '/detect/obstacle', '/detect/obstacle_side',
    '/detect/front_range', '/arx/armed_mission', '/cmd_vel', '/odom', '/scan',
    '/rosout',
)


class Recorder:
    """`ros2 bag record` in its own process group, so Ctrl+C reaches all of it."""

    def __init__(self):
        self.proc = None
        self.name = None        # folder of the run being or last recorded
        self.since = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        os.makedirs(RUNS, exist_ok=True)
        name = time.strftime('%m%d_%H%M')
        # Two runs in the same minute: the bag refuses a folder that exists.
        base, n = name, 2
        while os.path.exists(os.path.join(RUNS, name)):
            name, n = f'{base}_{n}', n + 1
        log = open(os.path.join(RUNS, name + '.record.log'), 'w')
        self.proc = subprocess.Popen(
            ['ros2', 'bag', 'record', '-o', os.path.join(RUNS, name), *RECORD_TOPICS],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.name, self.since = name, time.monotonic()

    def stop(self, wait=6.0):
        """SIGINT, as Ctrl+C would - the bag closes its file on it."""
        if not self.running():
            return
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
            self.proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def note(self, text):
        """A line in ~/runs/notes.txt: folder, time into the run, the text."""
        os.makedirs(RUNS, exist_ok=True)
        at = ''
        if self.since is not None and self.running():
            t = int(time.monotonic() - self.since)
            at = f'  (~{t // 60}:{t % 60:02d})'
        with open(os.path.join(RUNS, 'notes.txt'), 'a') as f:
            f.write(f'{self.name or "-"}{at}  {text}\n')


class Window:

    def __init__(self, node, with_image, target):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image as PilImage
        from PIL import ImageTk
        self.tk, self.pil, self.pil_tk = tk, PilImage, ImageTk
        self.node, self.with_image, self.target = node, with_image, target
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
            for i, (name, topic) in enumerate(IMAGES):
                rb = tk.Radiobutton(pick, text=name, value=topic, variable=self.which,
                                    command=self.switch, indicatoron=False, width=10,
                                    bg='#2a2a2a', fg=fg, selectcolor='#3a5a8a',
                                    activebackground='#3a3a3a', activeforeground=fg)
                rb.grid(row=0, column=i, padx=(0, 4))
                if topic == OBSTACLE_IMAGE:
                    self.obst_tab = rb
                if topic == SIGN_IMAGE:
                    # The last sign detect_sign reported, under its button,
                    # whichever image is being looked at.
                    self.sign_tab_text = tk.Label(
                        pick, text='-', bg='#1b1b1b', fg='#777', width=12,
                        font=('DejaVu Sans Mono', 9, 'bold'))
                    self.sign_tab_text.grid(row=1, column=i, padx=(0, 4), pady=(2, 0))
            self.img_info = tk.StringVar(value='')
            label(left, textvariable=self.img_info, fg=dim, anchor='w',
                  justify='left').pack(fill='x', pady=(4, 0))
            col = 1

        right = ttk.Frame(root, style='Dark.TFrame', padding=8)
        right.grid(row=0, column=col, sticky='n')

        # Record a run: one button, and a box for a line in ~/runs/notes.txt.
        self.rec = Recorder()
        rec = ttk.Frame(right, style='Dark.TFrame')
        rec.pack(fill='x', pady=(0, 8))
        self.rec_button = tk.Button(rec, width=16, font=('TkDefaultFont', 11, 'bold'),
                                    fg='white', activeforeground='white',
                                    command=self.toggle_record)
        self.rec_button.pack(side='left')
        self.rec_note = tk.Entry(rec, width=24, bg='#2a2a2a', fg=fg,
                                 insertbackground=fg)
        self.rec_note.pack(side='left', padx=(8, 4), fill='x', expand=True)
        self.rec_note.bind('<Return>', lambda e: self.add_note())
        tk.Button(rec, text='note', command=self.add_note, padx=4, pady=0,
                  bg='#2a2a2a', fg=fg, activebackground='#3a3a3a',
                  activeforeground=fg).pack(side='left')
        self.rec_text = label(right, text='', font=mono, fg=dim, anchor='w')
        self.rec_text.pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.paint_record()

        # Which mission is armed, and the last sign detect_sign reported. It
        # only reports a sign it matched, so this is the latest hit and how
        # long ago - not what is in front of the camera now.
        self.sign_text = label(right, text='-', font=mono, anchor='w', justify='left')
        self.sign_text.pack(fill='x')
        # detect_obstacle: what is in the boxes ahead, beside, and how far.
        # It only reports while construction, parking or the tunnel is armed.
        self.obst_text = label(right, text='-', font=mono, anchor='w', justify='left')
        self.obst_text.pack(fill='x', pady=(0, 8))

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

        label(right, text='speed  (/cmd_vel, odom)', fg=dim, anchor='w'
              ).pack(fill='x', pady=(10, 0))
        self.speed_text = label(right, text='-', font=mono, anchor='w')
        self.speed_text.pack(fill='x')
        self.speed_bar = tk.Canvas(right, width=BAR_W, height=16, bg='#111',
                                   highlightthickness=0)
        self.speed_bar.pack(anchor='w', pady=(2, 0))
        self.speed_fill = self.speed_bar.create_rectangle(0, 2, 0, 14, outline='')
        self.speed_odom = self.speed_bar.create_line(0, 0, 0, 16, width=3, fill='#e8e8e8')

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

        label(right, text='heading  (/odom yaw, + is left)', fg=dim, anchor='w'
              ).pack(fill='x', pady=(10, 0))
        row = ttk.Frame(right, style='Dark.TFrame')
        row.pack(fill='x')
        d = self.dial = tk.Canvas(row, width=DIAL, height=DIAL, bg='#1b1b1b',
                                  highlightthickness=0)
        d.pack(side='left')
        c, r = DIAL / 2, DIAL / 2 - 22
        d.create_oval(c - r, c - r, c + r, c + r, outline='#555', width=2)
        for deg, text in ((0, '0'), (90, '90'), (180, '180'), (-90, '-90')):
            x, y = self.dial_xy(deg, r + 12)
            d.create_text(x, y, text=text, fill=dim, font=('TkDefaultFont', 7))
        tx, ty = self.dial_xy(target, r)
        d.create_line(c, c, tx, ty, fill='#30b050', dash=(3, 3), width=2)
        self.dial_arrow = d.create_line(c, c, c, c + r, fill=fg, width=4,
                                        arrow='last', arrowshape=(12, 14, 5))
        self.yaw_text = label(row, text='-', font=('TkFixedFont', 16, 'bold'),
                              anchor='w', justify='left')
        self.yaw_text.pack(side='left', padx=(12, 0))

        foot = ttk.Frame(right, style='Dark.TFrame')
        foot.pack(fill='x', pady=(10, 0))
        self.footer = tk.StringVar(value='')
        label(foot, textvariable=self.footer, font=mono, fg=dim, anchor='w',
              justify='left').pack(side='left')
        self.batt_text = label(foot, text='', font=mono, fg=dim, anchor='e')
        tk.Button(foot, text='batt', command=self.show_battery, padx=4, pady=0,
                  font=('TkDefaultFont', 8), bg='#2a2a2a', fg=fg,
                  activebackground='#3a3a3a', activeforeground=fg
                  ).pack(side='right')
        self.batt_text.pack(side='right', padx=(0, 6))

        root.after(100, self.tick)

    def toggle_record(self):
        if self.rec.running():
            self.rec_text.config(text='stopping - closing the bag ...')
            self.root.update_idletasks()
            self.rec.stop()
            self.rec_text.config(text=f'saved ~/runs/{self.rec.name}')
        else:
            try:
                self.rec.start()
                self.rec_text.config(text=f'recording to ~/runs/{self.rec.name}')
            except OSError as e:
                self.rec_text.config(text=f'could not start: {e}')
        self.paint_record()

    def paint_record(self):
        if self.rec.running():
            t = int(time.monotonic() - self.rec.since)
            self.rec_button.config(text=f'\u25a0 STOP  {t // 60}:{t % 60:02d}',
                                   bg='#c0392b', activebackground='#e74c3c')
        else:
            self.rec_button.config(text='\u25cf REC', bg='#7b241c',
                                   activebackground='#a93226')
            if self.rec.proc is not None and self.rec.proc.poll() not in (None, 0, -2, 130):
                self.rec_text.config(
                    text=f'ros2 bag stopped by itself - see ~/runs/{self.rec.name}.record.log')

    def add_note(self):
        text = self.rec_note.get().strip()
        if not text:
            return
        self.rec.note(text)
        self.rec_note.delete(0, 'end')
        self.rec_text.config(text=f'noted for {self.rec.name or "-"}: {text}')

    def close(self):
        """Close the bag properly before the window goes."""
        self.rec.stop()
        self.root.destroy()

    def switch(self):
        if rclpy.ok():
            self.node.watch(self.which.get())
            self.shown = 0.0

    def show_battery(self):
        """The latest /battery_state, once - it is not kept up to date."""
        values = self.node.snapshot()[0]
        v = values.get('/battery_state')
        if v is None:
            self.batt_text.config(text='no /battery_state - robot.launch up?', fg='#777')
            return
        (volts, pct), stamp = v
        age = time.monotonic() - stamp
        # BatteryState says 0-1; some drivers send 0-100. Take either.
        pct = pct * 100 if pct <= 1.0 else pct
        colour = '#e04040' if volts < 11.0 else '#e0b000' if volts < 11.4 else '#30b050'
        self.batt_text.config(
            text=f'{volts:.2f} V  {pct:.0f}%  at {time.strftime("%H:%M:%S")}'
                 + (f'  ({age:.0f} s old)' if age > 5 else ''),
            fg=colour)

    @staticmethod
    def dial_xy(deg, radius):
        """0 points down, 180 up, + turns counter-clockwise, as yaw does.

        Down for 0 because the robot sets off down the start lane on the map,
        so the dial reads the same way round as docs/curve_map.png.
        """
        a = math.radians(deg)
        return DIAL / 2 + radius * math.sin(a), DIAL / 2 + radius * math.cos(a)

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
            ah, ah_stale = get('/detect/lane_ahead')
            ahead = '' if ah is None or ah_stale else f'   ahead {ah - CENTER:+5.0f}'
            colour = '#777' if stale else self.err_colour(e)
            self.err_text.config(
                text=f'{e:+7.1f} px   centre {c:6.1f}{ahead}'
                     + ('   (stale)' if stale else ''), fg=colour)
            x = mid + max(-1.0, min(1.0, e / CENTER)) * mid
            self.bar.coords(self.bar_fill, min(mid, x), 4, max(mid, x), BAR_H - 4)
            self.bar.itemconfig(self.bar_fill, fill=colour)
            self.bar.coords(self.bar_mark, x, 0, x, BAR_H)
        self.draw_plot(errors, now)

        # speed: what was commanded, and what odom says happened
        cmd, cmd_stale = get('/cmd_vel')
        ov, _ = get('/odom_v')
        if cmd is None:
            self.speed_text.config(text='no /cmd_vel yet', fg='#777')
            self.speed_bar.coords(self.speed_fill, 0, 2, 0, 14)
        else:
            v, w = cmd
            frac = max(0.0, min(1.0, v / WHEEL_MAX))
            colour = '#777' if cmd_stale else (
                '#30b050' if frac >= 0.85 else '#e0b000' if frac >= 0.5 else '#e04040')
            odom_txt = '' if ov is None else f'   odom {ov:5.3f}'
            self.speed_text.config(
                text=f'cmd {v:5.3f} m/s  w {w:+5.2f}{odom_txt}'
                     + ('   (stale)' if cmd_stale else ''), fg=colour)
            self.speed_bar.coords(self.speed_fill, 0, 2, frac * BAR_W, 14)
            self.speed_bar.itemconfig(self.speed_fill, fill=colour)
        if ov is not None:
            x = max(0.0, min(1.0, ov / WHEEL_MAX)) * BAR_W
            self.speed_bar.coords(self.speed_odom, x, 0, x, 16)

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

        # heading
        yaw, stale = get('/odom')
        if yaw is None:
            self.yaw_text.config(text='no /odom', fg='#777')
        else:
            x, y = self.dial_xy(yaw, DIAL / 2 - 26)
            self.dial.coords(self.dial_arrow, DIAL / 2, DIAL / 2, x, y)
            self.dial.itemconfig(self.dial_arrow, fill='#777' if stale else '#e8e8e8')
            to_go = (self.target - yaw + 180.0) % 360.0 - 180.0
            near = abs(to_go) <= 3.0
            self.yaw_text.config(
                text=f'{yaw:+7.1f} deg\nto {self.target:.0f}: {to_go:+6.1f}'
                     + ('\n(stale)' if stale else ''),
                fg='#777' if stale else '#30b050' if near else '#e8e8e8')

        lat, lat_stale = get('/detect/lane_latency_ms')
        lat_txt = '' if lat is None else (
            f'   latency {lat:4.0f} ms' + ('  (clocks?)' if lat < 0 or lat > 5000 else ''))
        self.footer.set(f'/detect/lane  {hz:4.1f} Hz{lat_txt}')

        # mission and sign
        m, m_stale = get('/arx/armed_mission')
        armed = '?' if m is None else MISSIONS.get(m, str(m))
        v = values.get('/detect/traffic_sign')
        if v is None:
            sign, colour = 'no sign yet', '#777'
        else:
            age = now - v[1]
            sign = f'{SIGNS.get(v[0], str(v[0]))} ({v[0]})  {age:.1f} s ago'
            colour = '#30b050' if age < 1.0 else '#e8e8e8' if age < 5.0 else '#777'
        self.sign_text.config(text=f'armed: {armed}\nsign:  {sign}', fg=colour)
        under = getattr(self, 'sign_tab_text', None)
        if under is not None:
            if v is None:
                under.config(text='-', fg='#777')
            else:
                # Short, so the label stays the width of the button above it.
                name = {1: 'CONSTR'}.get(v[0], SIGNS.get(v[0], str(v[0])))
                under.config(text=f'{name} {age:.0f}s', fg=colour)

        # obstacle
        b, b_stale = get('/detect/obstacle')
        sd, _ = get('/detect/obstacle_side')
        fr, _ = get('/detect/front_range')
        # The obstacle tab button: red while detect_obstacle is not looking
        # (it publishes nothing until mission_control arms construction,
        # parking or the tunnel, or obstacle.always_on), green once it is.
        tab = getattr(self, 'obst_tab', None)
        if tab is not None:
            live = b is not None and not b_stale
            tab.config(bg='#2f7d46' if live else '#8a2a2a',
                       selectcolor='#3fa05c' if live else '#b03a3a',
                       activebackground='#3a8a52' if live else '#9a3a3a')
        if b is None:
            self.obst_text.config(text='obstacle: no report - armed? /scan up?', fg='#777')
        else:
            ahead = ' + '.join(n for bit, n in BLOCKED_BITS if b & bit) or 'clear'
            side = '-' if sd is None else SIDES.get(sd, str(sd))
            rng = '-' if fr is None else 'none' if fr >= 9.0 else f'{fr:.2f} m'
            self.obst_text.config(
                text=f'obstacle: {ahead}   side: {side}   front: {rng}'
                     + ('  (stale)' if b_stale else ''),
                fg='#777' if b_stale else '#e04040' if b else '#30b050')

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
                    else 'no sign image - detect_sign only looks, and publishes\n'
                         'this, while a mission with a sign is armed'
                    if self.which.get() == SIGN_IMAGE
                    else 'no LiDAR image - detect_obstacle only publishes while\n'
                         'construction, parking or the tunnel is armed'
                    if self.which.get() == OBSTACLE_IMAGE
                    else 'no mask - lane.publish_debug_image: true\n'
                         'in lane.yaml, then restart detect_lane')

        self.paint_record()
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
    ap.add_argument('--target', type=float, default=180.0,
                    help='heading to mark on the dial, deg (default 180)')
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
        Window(node, with_image=not args.no_image, target=args.target).run()
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

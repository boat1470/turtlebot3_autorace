#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Measure a box on the camera image, and turn the camera's knobs, live.

    python3 calibrate_ui/roi_probe.py                         a window, /camera/image_raw
    python3 calibrate_ui/roi_probe.py /camera/image_projected
    python3 calibrate_ui/roi_probe.py --size 200              start with a bigger box
    python3 calibrate_ui/roi_probe.py --cli                   no window, one line per 10 frames

Several calibration steps need a number off the image rather than a look at
it, and rqt_image_view gives no numbers. The window shows the image with a
box on it - drag inside to move it, drag the corner to resize it, drag
anywhere else to draw a new one, or nudge it a pixel at a time with the -/+
buttons - and measures inside the box ten times a second:

    mean B G R   what the colour balance is doing
    max          the brightest value. 255 means clipped: set exposure and gain
                 so white paper sits around 200-235
    H S V        5th to 95th percentile, OpenCV units (H 0-179). For thresholds
    sharp        variance of the Laplacian. Turn the lens for the highest number
    x R  x B     with white paper in the box, what to multiply the red and blue
                 ColourGains by to bring them level with green

Beside it are the camera controls this calibration moves - ExposureTime,
AnalogueGain, Sharpness and ColourGains - as sliders with -/+ buttons. Moving
one sets it on the running camera node with `ros2 param set` semantics, and
the value shown is read back from the node afterwards, not assumed. Two
things the sliders do for you, because getting them wrong looks like the
slider not working:

  * ExposureTime and AnalogueGain only take effect with ExposureTimeMode and
    AnalogueGainMode at 1 (manual; 0 is auto), so those are switched first.
    ColourGains only takes effect with AwbEnable false, so that is switched
    first too.
  * camera_ros applies parameters on the next frame and drops any set that a
    second set overtakes before then, so sets go out one at a time, at most
    one per 150 ms, and a slider being dragged only sends where it is now.

The mode goes out on its own before ExposureTime, not in the same request.
Unpatched camera_ros (see file_in_pi/camera_ros_patches/) can still be
holding its default 20000 from startup, and a request carrying
ExposureTimeMode is the one that releases it - on top of whatever
ExposureTime the same request carried.

The bottom line is the launch command for what the camera holds now, to copy
into the notes once the numbers are right. ColourGains has no launch argument
yet, so it is printed as a param set to run after.

Raw image topics only - not /compressed. Frames are always converted to bgr8,
so this cannot check what encoding the camera sends; `ros2 topic echo
/camera/image_raw --once --field encoding` does that.
"""

import argparse
import collections
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter import parameter_value_to_python
from sensor_msgs.msg import Image

# key, slider min, max, step of the slider and of the -/+ buttons
SLIDERS = (
    ('ExposureTime', 100, 33000, 100),      # us; 33000 is one frame at 30 fps
    ('AnalogueGain', 1.0, 16.0, 0.05),
    ('Sharpness', 0.0, 16.0, 0.1),
    ('ColourGains red', 0.5, 4.0, 0.01),
    ('ColourGains blue', 0.5, 4.0, 0.01),
)

# What is read back from the camera after every set.
READ = ('ExposureTime', 'ExposureTimeMode', 'AnalogueGain', 'AnalogueGainMode',
        'AwbEnable', 'ColourGains', 'Sharpness')

SEND_GAP = 0.15     # s between two sets, so camera_ros does not drop one
SEND_TIMEOUT = 2.0  # s before a set with no answer is given up on
VIEW = (640, 480)   # the image is scaled to fit this, up or down


def measure(roi):
    px = roi.reshape(-1, 3)
    b, g, r = px.mean(axis=0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return {
        'b': b, 'g': g, 'r': r, 'max': int(roi.max()),
        'lo': np.percentile(hsv, 5, axis=0), 'hi': np.percentile(hsv, 95, axis=0),
        'sharp': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'xr': g / max(r, 1.0), 'xb': g / max(b, 1.0),
    }


def one_line(m):
    lo, hi = m['lo'], m['hi']
    return (f'BGR {m["b"]:5.1f} {m["g"]:5.1f} {m["r"]:5.1f}  max {m["max"]:3d} | '
            f'H {lo[0]:.0f}-{hi[0]:.0f} S {lo[1]:.0f}-{hi[1]:.0f} '
            f'V {lo[2]:.0f}-{hi[2]:.0f} | sharp {m["sharp"]:7.1f} | '
            f'x R {m["xr"]:.3f} x B {m["xb"]:.3f}')


def to_param(name, value):
    if isinstance(value, bool):
        kind = Parameter.Type.BOOL
    elif isinstance(value, int):
        kind = Parameter.Type.INTEGER
    elif isinstance(value, float):
        kind = Parameter.Type.DOUBLE
    else:
        kind, value = Parameter.Type.DOUBLE_ARRAY, [float(v) for v in value]
    return Parameter(name, kind, value).to_parameter_msg()


class Probe(Node):
    """Keeps the latest frame and talks to the camera's parameter services.

    Runs in its own thread. Answers are left on `results` for the window to
    pick up, because Tk may only be touched from the thread that made it.
    """

    def __init__(self, topic, camera, cli_size=None):
        super().__init__('roi_probe')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frame, self.stamp = None, 0.0
        self.cli_size, self.count = cli_size, 0
        self.results = collections.deque()
        self.create_subscription(Image, topic, self.on_image, 1)
        self.set_cli = self.create_client(SetParameters, f'{camera}/set_parameters')
        self.get_cli = self.create_client(GetParameters, f'{camera}/get_parameters')

    def on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            self.frame, self.stamp = img, time.monotonic()
        if self.cli_size:
            self.count += 1
            if self.count % 10 == 0:
                h, w = img.shape[:2]
                s = self.cli_size // 2
                print(one_line(measure(img[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s])),
                      flush=True)

    def latest(self):
        with self.lock:
            return self.frame, self.stamp

    def set(self, values):
        req = SetParameters.Request()
        req.parameters = [to_param(k, v) for k, v in values.items()]
        self.set_cli.call_async(req).add_done_callback(
            lambda f: self.results.append(('set', values, f.result())))

    def get(self, names):
        req = GetParameters.Request()
        req.names = list(names)
        self.get_cli.call_async(req).add_done_callback(
            lambda f: self.results.append(('get', names, f.result())))


class Window:

    def __init__(self, node, size, camera):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image as PilImage
        from PIL import ImageTk
        self.tk, self.pil, self.pil_tk = tk, PilImage, ImageTk

        self.node, self.size, self.camera = node, size, camera
        self.roi = None             # [x, y, w, h] in image pixels
        self.img_wh = None
        self.scale = 1.0
        self.drag = None
        self.photo = None
        self.shown = 0.0
        self.current = {}           # what the camera last said it holds
        self.queue = collections.OrderedDict()   # key -> values; newest wins
        self.in_flight = None       # time the outstanding set went out
        self.last_sent = 0.0
        self.last_try = 0.0         # last read attempted before the camera answered
        self.loading = False        # true while sliders are moved by a read-back

        root = self.root = tk.Tk()
        root.title('arx roi probe')
        mono = ('TkFixedFont', 10)

        self.canvas = tk.Canvas(root, width=VIEW[0], height=VIEW[1],
                                bg='black', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8, sticky='n')
        self.img_item = self.canvas.create_image(0, 0, anchor='nw')
        self.box = self.canvas.create_rectangle(0, 0, 0, 0, outline='#00ff66', width=2)
        self.handle = self.canvas.create_rectangle(0, 0, 0, 0, fill='#00ff66', outline='')
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'drag', None))

        side = ttk.Frame(root, padding=(0, 8, 8, 8))
        side.grid(row=0, column=1, sticky='n')

        self.stats = tk.StringVar(value='waiting for an image...')
        tk.Label(side, textvariable=self.stats, font=mono, justify='left',
                 anchor='w', width=46, bg='#111', fg='#ddd', padx=8, pady=6
                 ).pack(fill='x')

        def stepper(parent, row, text, minus, plus):
            ttk.Label(parent, text=text, width=16).grid(row=row, column=0, sticky='w')
            for col, (label, cmd) in ((1, ('-', minus)), (3, ('+', plus))):
                tk.Button(parent, text=label, width=2, command=cmd,
                          repeatdelay=350, repeatinterval=60).grid(row=row, column=col)

        box = ttk.LabelFrame(side, text='box (px, -/+ by 1)', padding=6)
        box.pack(fill='x', pady=(8, 0))
        self.roi_vars = []
        for i, key in enumerate('xywh'):
            var = tk.StringVar(value='-')
            self.roi_vars.append(var)
            stepper(box, i, key, lambda i=i: self.nudge(i, -1), lambda i=i: self.nudge(i, +1))
            ttk.Label(box, textvariable=var, width=6, anchor='center').grid(row=i, column=2)

        cam = ttk.LabelFrame(side, text=f'camera {camera}', padding=6)
        cam.pack(fill='x', pady=(8, 0))
        self.vars = {}
        for i, (key, lo, hi, step) in enumerate(SLIDERS):
            # ColourGains starts at 1.5, not the slider's floor: camera_ros
            # cannot give it a default, so until it is set the camera reads
            # back nothing, and "from box" would otherwise scale up from 0.5.
            start = 1.5 if key.startswith('ColourGains') else lo
            var = self.vars[key] = tk.DoubleVar(value=start)
            stepper(cam, i, key,
                    lambda k=key, s=step: self.bump(k, -s),
                    lambda k=key, s=step: self.bump(k, +s))
            tk.Scale(cam, variable=var, from_=lo, to=hi, resolution=step,
                     orient='horizontal', length=200, showvalue=True,
                     command=lambda _v, k=key: self.moved(k)
                     ).grid(row=i, column=2, padx=4)

        self.awb = tk.BooleanVar(value=True)
        row = ttk.Frame(cam)
        row.grid(row=len(SLIDERS), column=0, columnspan=4, sticky='w', pady=(4, 0))
        ttk.Checkbutton(row, text='AwbEnable (auto white balance)', variable=self.awb,
                        command=self.awb_clicked).pack(side='left')
        self.modes = tk.StringVar(value='')
        ttk.Label(cam, textvariable=self.modes, foreground='#555').grid(
            row=len(SLIDERS) + 1, column=0, columnspan=4, sticky='w')

        bar = ttk.Frame(side)
        bar.pack(fill='x', pady=(8, 0))
        ttk.Button(bar, text='ColourGains from box (white paper)',
                   command=self.white_from_box).pack(side='left')
        ttk.Button(bar, text='Read camera', command=self.read).pack(side='left', padx=6)

        self.cmd = tk.StringVar(value='')
        low = ttk.Frame(root, padding=(8, 0, 8, 8))
        low.grid(row=2, column=0, columnspan=2, sticky='we')
        ttk.Entry(low, textvariable=self.cmd, font=mono, state='readonly'
                  ).pack(side='left', fill='x', expand=True)
        ttk.Button(low, text='Copy', command=self.copy).pack(side='left', padx=(6, 0))
        self.status = tk.StringVar(value='')
        ttk.Label(root, textvariable=self.status, foreground='#555',
                  padding=(8, 0, 8, 8)).grid(row=3, column=0, columnspan=2, sticky='w')

        self.read()
        root.after(50, self.tick)

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
        self.draw_box()

    def nudge(self, i, d):
        if self.roi is None:
            return
        self.roi[i] += d
        self.clamp()
        self.draw_box()

    def draw_box(self):
        x, y, w, h = self.roi
        s = self.scale
        self.canvas.coords(self.box, x * s, y * s, (x + w) * s, (y + h) * s)
        self.canvas.coords(self.handle, (x + w) * s - 5, (y + h) * s - 5,
                           (x + w) * s + 5, (y + h) * s + 5)
        for var, v in zip(self.roi_vars, self.roi):
            var.set(str(v))

    # --- the camera ----------------------------------------------------

    def moved(self, key):
        if not self.loading:
            self.send(key)

    def bump(self, key, d):
        lo, hi = next((lo, hi) for k, lo, hi, _ in SLIDERS if k == key)
        var = self.vars[key]
        var.set(round(min(hi, max(lo, var.get() + d)), 4))
        self.send(key)

    def send(self, key):
        v = self.vars
        if key == 'ExposureTime':
            if self.current.get('ExposureTimeMode') != 1:
                self.queue['ExposureTimeMode'] = {'ExposureTimeMode': 1}
            self.queue['ExposureTime'] = {'ExposureTime': int(v[key].get())}
        elif key == 'AnalogueGain':
            self.queue['AnalogueGain'] = {'AnalogueGainMode': 1,
                                          'AnalogueGain': float(v[key].get())}
        elif key == 'Sharpness':
            self.queue['Sharpness'] = {'Sharpness': float(v[key].get())}
        else:
            if self.current.get('AwbEnable') is not False:
                self.queue['AwbEnable'] = {'AwbEnable': False}
            self.queue['ColourGains'] = {'ColourGains': [
                float(v['ColourGains red'].get()), float(v['ColourGains blue'].get())]}

    def awb_clicked(self):
        self.queue['AwbEnable'] = {'AwbEnable': bool(self.awb.get())}

    def white_from_box(self):
        m = self.last_measure
        if m is None:
            return
        for key, x in (('ColourGains red', m['xr']), ('ColourGains blue', m['xb'])):
            self.vars[key].set(round(self.vars[key].get() * x, 3))
        self.send('ColourGains red')
        self.status.set(f'ColourGains x R {m["xr"]:.3f} x B {m["xb"]:.3f} - '
                        'measure again, two or three rounds settle it')

    def read(self):
        if not self.node.get_cli.service_is_ready():
            self.status.set(f'no {self.camera}/get_parameters - is the camera '
                            'running, and ROS_DOMAIN_ID the same as the Pi?')
            return
        self.node.get(READ)

    def took(self, names, res):
        if res is None:
            self.status.set('read failed')
            return
        for name, val in zip(names, res.values):
            self.current[name] = parameter_value_to_python(val)
        c = self.current
        self.loading = True
        for key in ('ExposureTime', 'AnalogueGain', 'Sharpness'):
            if c.get(key) is not None:
                self.vars[key].set(c[key])
        if isinstance(c.get('ColourGains'), (list, tuple)) and len(c['ColourGains']) == 2:
            self.vars['ColourGains red'].set(round(c['ColourGains'][0], 3))
            self.vars['ColourGains blue'].set(round(c['ColourGains'][1], 3))
        if c.get('AwbEnable') is not None:
            self.awb.set(bool(c['AwbEnable']))
        self.root.update_idletasks()
        self.loading = False

        def mode(name):
            return {0: '0 auto', 1: '1 manual'}.get(c.get(name), str(c.get(name)))
        cg = c.get('ColourGains')
        unset = '' if isinstance(cg, (list, tuple)) and len(cg) == 2 else \
            '\nColourGains not set on the camera - sliders show a start point'
        self.modes.set(f'ExposureTimeMode {mode("ExposureTimeMode")}   '
                       f'AnalogueGainMode {mode("AnalogueGainMode")}{unset}')
        self.update_cmd()

    def update_cmd(self):
        c = self.current
        cmd = 'ros2 launch arx_camera_bringup camera.launch.py ae:=false'
        if c.get('ExposureTime') is not None:
            cmd += f' exposure_us:={int(c["ExposureTime"])}'
        if c.get('AnalogueGain') is not None:
            cmd += f' gain:={c["AnalogueGain"]:.2f}'
        cmd += f' awb:={"true" if c.get("AwbEnable") else "false"}'
        if c.get('Sharpness') is not None:
            cmd += f' sharpness:={c["Sharpness"]:.1f}'
        cg = c.get('ColourGains')
        if isinstance(cg, (list, tuple)) and len(cg) == 2:
            cmd += (f'   # then: ros2 param set {self.camera} ColourGains '
                    f'"[{cg[0]:.3f}, {cg[1]:.3f}]"')
        self.cmd.set(cmd)

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.cmd.get())
        self.status.set('copied')

    # --- the loop ------------------------------------------------------

    last_measure = None

    def tick(self):
        now = time.monotonic()

        while self.node.results:
            kind, what, res = self.node.results.popleft()
            if kind == 'get':
                self.took(what, res)
                continue
            self.in_flight = None
            bad = [] if res is None else [r.reason for r in res.results if not r.successful]
            if res is None or bad:
                self.status.set(f'{", ".join(what)}: {"; ".join(bad) or "no answer"}')
            else:
                self.status.set('set ' + ', '.join(f'{k}={v}' for k, v in what.items()))
            self.node.get(READ)

        if self.in_flight and now - self.in_flight > SEND_TIMEOUT:
            self.status.set('camera did not answer a set')
            self.in_flight = None
        if (self.queue and self.in_flight is None and now - self.last_sent >= SEND_GAP
                and self.node.set_cli.service_is_ready()):
            _, values = self.queue.popitem(last=False)
            self.node.set(values)
            self.in_flight = self.last_sent = now

        # Discovery is slow over Wi-Fi, and the window may be opened before
        # the camera is: keep asking once a second until it has answered.
        if not self.current and now - self.last_try > 1.0:
            self.last_try = now
            self.read()

        frame, stamp = self.node.latest()
        if frame is not None and stamp != self.shown:
            self.shown = stamp
            self.show(frame)

        self.root.after(50 if self.queue or self.in_flight else 100, self.tick)

    def show(self, frame):
        h, w = frame.shape[:2]
        if self.img_wh != (w, h):
            self.img_wh = (w, h)
            self.scale = min(VIEW[0] / w, VIEW[1] / h)
            self.canvas.config(width=int(w * self.scale), height=int(h * self.scale))
            s = min(self.size, w, h)
            self.roi = [(w - s) // 2, (h - s) // 2, s, s]
            self.draw_box()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.pil.fromarray(rgb).resize(
            (int(w * self.scale), int(h * self.scale)), self.pil.NEAREST)
        self.photo = self.pil_tk.PhotoImage(img)
        self.canvas.itemconfig(self.img_item, image=self.photo)

        x, y, bw, bh = self.roi
        m = self.last_measure = measure(frame[y:y + bh, x:x + bw])
        lo, hi = m['lo'], m['hi']
        clip = '   CLIPPED' if m['max'] >= 255 else ''
        self.stats.set(
            f'{w}x{h}   box {bw}x{bh} at ({x}, {y})\n'
            f'mean  B {m["b"]:5.1f}  G {m["g"]:5.1f}  R {m["r"]:5.1f}\n'
            f'max   {m["max"]:3d}{clip}\n'
            f'H {lo[0]:3.0f}-{hi[0]:3.0f}  S {lo[1]:3.0f}-{hi[1]:3.0f}  '
            f'V {lo[2]:3.0f}-{hi[2]:3.0f}   (p5-p95)\n'
            f'sharp {m["sharp"]:8.1f}\n'
            f'white paper: ColourGains x R {m["xr"]:.3f}  x B {m["xb"]:.3f}')

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('topic', nargs='?', default='/camera/image_raw')
    ap.add_argument('--size', type=int, default=40, help='starting box size, px')
    ap.add_argument('--camera', default='/camera', help='the camera_ros node')
    ap.add_argument('--cli', action='store_true', help='print lines, no window')
    args = ap.parse_args()

    rclpy.init()
    node = Probe(args.topic, args.camera, cli_size=args.size if args.cli else None)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = None
    try:
        if args.cli:
            executor.spin()
            return
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        Window(node, args.size, args.camera).run()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the spinning thread, and wait for it, before the node goes:
        # a thread still inside spin() when the interpreter exits aborts the
        # process ("terminate called without an active exception").
        executor.shutdown(timeout_sec=1.0)
        if spinner is not None:
            spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

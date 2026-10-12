#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Cut a reference sign out of the live camera and save it where detect_sign looks.

    python3 calibrate_ui/sign_capture.py
    python3 calibrate_ui/sign_capture.py --dir notes/ref_real

The window shows what detect_sign sees: /camera/image_compensated when it is
being published, otherwise /camera/image_rect_color with the same contrast
stretch image_compensation applies (clip_hist_percent 1.0) done here - the
source in use is printed under the image.

    drag              draw a box round the sign
    drag inside it    move it
    Freeze / space    hold the frame, so the box can be set on a still image
    Save              write the box, at camera resolution, to <dir>/<name>

Beside it, the box's size and how many SIFT descriptors it holds - the number
detect_sign's score is divided by. The arrows that work were cut with the disc
55-91 px across and 29-32 descriptors (arx_mission/image/README.md); a sign
much smaller than that in frame wants the robot moved closer before cutting.

The default directory is arx_mission/image, the package's own, which is
installed as symlinks: overwriting one of the names detect_sign already loads
(construction.png, parking.png, left.png, right.png) takes effect on its next
start without a build. A new name needs `colcon build` and an entry in
sign.yaml. Saving over a file asks first. The whole frame is saved beside the
cut as <name>_frame.png under <dir>/frames/, for grading the reference on
frames it was not cut from later.
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.normpath(os.path.join(HERE, '..', 'arx_mission', 'image'))
SCALE = 2           # 320x240 shown at 640x480
FRESH = 1.0         # s a compensated frame stays preferred over the fallback


def compensate(img, clip=1.0):
    """turtlebot3_autorace_camera/image_compensation.py, same arithmetic."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    acc = np.cumsum(cv2.calcHist([gray], [0], None, [256], [0, 256]))
    total = acc[255]
    c = clip * total / 100. / 2.
    lo = 0
    while acc[lo] < c:
        lo += 1
    hi = 255
    while acc[hi] >= total - c:
        hi -= 1
    if hi <= lo:
        return img
    a = 255.0 / (hi - lo)
    return cv2.convertScaleAbs(img, -1, a, -lo * a)


class Grabber(Node):
    """Latest frame from either source; spun in its own thread."""

    def __init__(self):
        super().__init__('sign_capture')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.comp = (None, 0.0)
        self.rect = (None, 0.0)
        self.create_subscription(Image, '/camera/image_compensated', self.on_comp, 1)
        self.create_subscription(CompressedImage, '/camera/image_rect_color/compressed',
                                 self.on_rect, 1)

    def on_comp(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            self.comp = (img, time.monotonic())

    def on_rect(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with self.lock:
                self.rect = (img, time.monotonic())

    def latest(self):
        """(frame, source name, stamp), compensated preferred while fresh."""
        with self.lock:
            (c, tc), (r, tr) = self.comp, self.rect
        if c is not None and time.monotonic() - tc < FRESH:
            return c, '/camera/image_compensated', tc
        if r is not None:
            return compensate(r), '/camera/image_rect_color + compensation here', tr
        return None, 'waiting for /camera/image_compensated or image_rect_color', 0.0


class Window:

    def __init__(self, node, folder):
        import tkinter as tk
        from tkinter import messagebox
        from tkinter import ttk
        from PIL import Image as PilImage
        from PIL import ImageTk
        self.tk, self.msg, self.pil, self.pil_tk = tk, messagebox, PilImage, ImageTk
        self.node, self.folder = node, folder
        self.sift = cv2.SIFT_create()
        self.frame, self.shown, self.photo = None, 0.0, None
        self.roi = None             # [x, y, w, h] in camera pixels
        self.drag = None
        self.frozen = False
        self.saved_at = 0.0          # keeps the saved message up a while

        root = self.root = tk.Tk()
        root.title('arx sign capture')
        mono = ('TkFixedFont', 10)

        self.canvas = tk.Canvas(root, width=320 * SCALE, height=240 * SCALE, bg='black',
                                highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8, sticky='n')
        self.img_item = self.canvas.create_image(0, 0, anchor='nw')
        self.box = self.canvas.create_rectangle(0, 0, 0, 0, outline='#00ff66', width=2)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'drag', None))

        side = ttk.Frame(root, padding=(0, 8, 8, 8))
        side.grid(row=0, column=1, sticky='n')

        self.freeze_btn = ttk.Button(side, text='Freeze  (space)', command=self.toggle_freeze)
        self.freeze_btn.pack(fill='x')
        root.bind('<space>', lambda e: self.toggle_freeze())

        self.info = tk.StringVar(value='draw a box round the sign')
        tk.Label(side, textvariable=self.info, font=mono, justify='left', anchor='w',
                 width=34, bg='#111', fg='#ddd', padx=8, pady=6).pack(fill='x', pady=(8, 0))

        self.preview = tk.Label(side, bg='#111')
        self.preview.pack(pady=(8, 0))

        ttk.Label(side, text='file name').pack(anchor='w', pady=(12, 0))
        self.name = tk.StringVar(value='construction.png')
        ttk.Entry(side, textvariable=self.name, width=30).pack(fill='x')
        ttk.Label(side, text=f'into {folder}', foreground='#555', wraplength=260
                  ).pack(anchor='w')
        ttk.Button(side, text='Save', command=self.save).pack(fill='x', pady=(8, 0))

        self.status = tk.StringVar(value='')
        ttk.Label(root, textvariable=self.status, foreground='#555',
                  padding=(8, 0, 8, 8)).grid(row=2, column=0, columnspan=2, sticky='w')
        root.after(100, self.tick)

    # --- the box ---------------------------------------------------------

    def press(self, e):
        ix, iy = e.x / SCALE, e.y / SCALE
        if self.roi:
            x, y, w, h = self.roi
            if x <= ix < x + w and y <= iy < y + h:
                self.drag = ('move', ix - x, iy - y)
                return
        self.drag = ('new', ix, iy)
        self.roi = [int(ix), int(iy), 1, 1]

    def motion(self, e):
        if not self.drag or self.frame is None:
            return
        H, W = self.frame.shape[:2]
        ix = min(max(e.x / SCALE, 0), W - 1)
        iy = min(max(e.y / SCALE, 0), H - 1)
        kind, a, b = self.drag
        if kind == 'move':
            w, h = self.roi[2], self.roi[3]
            self.roi = [int(min(max(ix - a, 0), W - w)), int(min(max(iy - b, 0), H - h)), w, h]
        else:
            x0, x1 = sorted((a, ix))
            y0, y1 = sorted((b, iy))
            self.roi = [int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0))]
        self.update_box()

    def update_box(self):
        x, y, w, h = self.roi
        self.canvas.coords(self.box, x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE)
        self.measure()

    def crop(self):
        if self.roi is None or self.frame is None:
            return None
        x, y, w, h = self.roi
        if w < 4 or h < 4:
            return None
        return self.frame[y:y + h, x:x + w]

    def measure(self):
        c = self.crop()
        if c is None:
            self.info.set('draw a box round the sign')
            self.preview.config(image='')
            return
        kp, des = self.sift.detectAndCompute(c, None)
        n = 0 if des is None else len(des)
        x, y, w, h = self.roi
        hint = ('  fine' if n >= 25 else '  thin - move closer?' if n >= 12
                else '  too few - move closer')
        self.info.set(f'box {w} x {h} px at ({x}, {y})\n'
                      f'SIFT descriptors {n}{hint}\n'
                      f'(the arrows that work: 29-32)')
        big = cv2.drawKeypoints(c, kp, None, color=(0, 255, 0))
        k = max(1, min(4, 220 // max(w, h)))
        big = cv2.resize(big, (w * k, h * k), interpolation=cv2.INTER_NEAREST)
        self.prev_photo = self.pil_tk.PhotoImage(
            self.pil.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
        self.preview.config(image=self.prev_photo)

    # --- freeze and save -------------------------------------------------

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self.freeze_btn.config(text='Live  (space)' if self.frozen else 'Freeze  (space)')
        self.status.set('frozen - set the box, then Save' if self.frozen else 'live')

    def save(self):
        c = self.crop()
        if c is None:
            self.status.set('nothing to save - draw a box first')
            return
        name = self.name.get().strip()
        if not name:
            self.status.set('give it a file name')
            return
        if not os.path.splitext(name)[1]:
            name += '.png'
        path = os.path.join(self.folder, name)
        if os.path.exists(path) and not self.msg.askyesno(
                'overwrite?', f'{path}\nalready exists. Save over it?'):
            self.status.set('not saved')
            return
        os.makedirs(os.path.join(self.folder, 'frames'), exist_ok=True)
        cv2.imwrite(path, c)
        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(self.folder, 'frames', f'{stem}_frame.png'), self.frame)
        x, y, w, h = self.roi
        self.saved_at = time.monotonic()
        self.status.set(f'saved {path}  ({w}x{h} px from ({x}, {y}))  '
                        f'+ frames/{stem}_frame.png')

    # --- the loop --------------------------------------------------------

    def tick(self):
        if not rclpy.ok():
            self.root.destroy()
            return
        if not self.frozen:
            frame, source, stamp = self.node.latest()
            if frame is not None and stamp != self.shown:
                self.shown, self.frame = stamp, frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                self.canvas.config(width=w * SCALE, height=h * SCALE)
                self.photo = self.pil_tk.PhotoImage(
                    self.pil.fromarray(rgb).resize((w * SCALE, h * SCALE), self.pil.NEAREST))
                self.canvas.itemconfig(self.img_item, image=self.photo)
                if self.roi:
                    self.measure()
            if time.monotonic() - self.saved_at > 6.0:
                self.status.set(source)
        self.root.after(100, self.tick)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dir', default=DEFAULT_DIR, help=f'where to save (default {DEFAULT_DIR})')
    args = ap.parse_args()
    folder = os.path.abspath(args.dir)

    rclpy.init()
    node = Grabber()
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
        Window(node, folder).run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=1.0)
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Start and stop the robot and the camera calibration from buttons.

    python3 calibrate_ui/launcher.py
    python3 calibrate_ui/launcher.py --host iotcentral@192.168.0.101

Two groups of buttons, one per command:

  Pi           camera and robot bringup, each on its own button, run on the
               Pi over ssh
  calibration  intrinsic, extrinsic and rqt_reconfigure, run here. Button 1
               (or the 1 key) starts all three at once, and stops them all
               while any is running

Each row has a start/stop button and a status that says whether it is
running or how it ended. The output of everything goes to the pane at the
bottom, each line tagged with the row it came from. Closing the window stops
everything still running.

Here, the commands start with this process's environment, so run it from a
terminal that has the workspace sourced. Stop sends Ctrl+C (SIGINT) to the
whole process group, which is what ros2 launch needs to bring its nodes down
cleanly, and kills after five seconds.

On the Pi, each runs as `ssh -tt <host> bash -ic '<command>'`:

  * bash -i, because the Pi's ~/.bashrc is where ROS, the overlays,
    ROS_DOMAIN_ID, TURTLEBOT3_MODEL and LDS_MODEL are set, and Ubuntu's
    .bashrc returns straight away for a shell that is not interactive.
    Sourcing it takes about 10 s on the Pi, so output starts late - and a
    Stop pressed in those 10 s interrupts the sourcing, not the launch.
  * -tt, so the command has a terminal on the Pi: Stop writes Ctrl+C into
    it, which reaches ros2 launch the same as pressing it in an ssh window.
    Signalling the local ssh would only end the connection.

  If the launch has not stopped five seconds after Ctrl+C, `pkill -INT -f`
  is sent on the Pi for the launch file by name, and the connection is
  closed five seconds after that. ssh has to log in without a password
  (BatchMode): set up a key with ssh-copy-id first.

sign board  the junction sign is an ESP32 with two arms, its own MQTT
            broker at --mqtt-host (192.168.0.102:1883). Pick left or right
            and a delay; with active on, every START of full.launch sends
            `down` at once and the chosen side once the delay is up. Turning
            active on sends `down` too. The board is asked for its state
            every 10 s. A command that cannot go out is said so in red -
            never dropped quietly.

                topic             way           payload
                /control/sign     PC -> board   left / right / down
                /cmd/state        PC -> board   anything = "what are you?"
                /response/state   board -> PC   left / right / none
"""

import argparse
import collections
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time

HOST = 'iotcentral@192.168.0.101'

# group, tag for the log, name, command
COMMANDS = (
    ('pi', 'cam', 'camera',
     'ros2 launch arx_camera_bringup camera.launch.py'),
    ('pi', 'robot', 'robot bringup',
     'ros2 launch turtlebot3_bringup robot.launch.py'),
    ('calibration', 'intr', 'intrinsic calibration',
     'ros2 launch turtlebot3_autorace_camera intrinsic_camera_calibration.launch.py'),
    ('calibration', 'extr', 'extrinsic calibration',
     'ros2 launch turtlebot3_autorace_camera extrinsic_camera_calibration.launch.py '
     'calibration_mode:=True'),
    ('calibration', 'rqt', 'rqt_reconfigure',
     'ros2 run rqt_reconfigure rqt_reconfigure'),
)

# The full stack on this PC, started at whichever of full.launch.py's presets
# is picked. The list is read from the launch file, so a new preset shows up
# without editing this one.
FULL_LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                           'arx_mission', 'launch', 'full.launch.py')
MISSION_CMD = 'ros2 launch arx_mission full.launch.py sim:=false start:={}'


def presets():
    try:
        with open(FULL_LAUNCH) as f:
            names = re.findall(r"^    '(\w+)': \{\n        'stage'", f.read(), re.M)
    except OSError:
        names = []
    return names or ['full', 'mission3', 'mission4', 'mission5', 'mission6']


# A line that means the command is up, not just started. Until one turns up
# the row's light is red ("starting..."); after it, green ("ready"). A command
# with none here goes green as soon as it is running - there is nothing to
# wait for that the launcher can see.
READY = {
    # turtlebot3_node's last line once the motors and odometry are live.
    'robot': re.compile(r'\[diff_drive_controller\]: Run!'),
}

STOP_WAIT = 5.0     # s after each way of stopping before trying the next
LOG_LINES = 2000    # kept in the output pane


class Job:
    """One command run here, and the process running it, if any."""

    def __init__(self, tag, name, command, out):
        self.tag, self.name, self.command, self.out = tag, name, command, out
        self.argv = shlex.split(command)
        self.shown = command                # what the window prints beside it
        self.proc = None
        self.stop_sent = None
        self.ready_re = READY.get(tag)
        self.ready = False

    def say(self, text):
        self.out.append(f'[{self.tag}] {text}')

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def popen(self, stdin):
        try:
            # A session of its own, so a signal reaches ros2 launch and every
            # node it started, and Ctrl+C in this terminal does not.
            self.proc = subprocess.Popen(
                self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=stdin, start_new_session=True, text=True, bufsize=1)
        except FileNotFoundError as e:
            self.say(f'{e} - is the workspace sourced?')
            self.proc = None
            return
        self.stop_sent = None
        self.ready = False
        threading.Thread(target=self.pump, args=(self.proc,), daemon=True).start()

    def start(self):
        if self.running():
            return
        self.say(f'$ {self.command}')
        self.popen(subprocess.DEVNULL)

    def pump(self, proc):
        for line in proc.stdout:
            if self.ready_re is not None and not self.ready and self.ready_re.search(line):
                self.ready = True
            self.say(line.rstrip())

    def signal(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def stop(self):
        if not self.running() or self.stop_sent is not None:
            return
        self.say('stopping (Ctrl+C)')
        self.signal(signal.SIGINT)
        self.stop_sent = time.monotonic()

    def tick(self):
        """Kill a group that ignored Ctrl+C for too long."""
        
        if (self.running() and self.stop_sent is not None
                and time.monotonic() - self.stop_sent > STOP_WAIT):
            self.say('did not stop - killing')
            self.signal(signal.SIGKILL)
            self.stop_sent = time.monotonic()

    def status(self):
        if self.proc is None:
            return 'stopped', '#777'
        code = self.proc.poll()
        if code is None:
            if self.stop_sent:
                return 'stopping...', '#e0b000'
            if self.ready_re is not None:
                return ('ready', '#30b050') if self.ready else ('starting...', '#e04040')
            return 'running', '#30b050'
        if code == 0 or self.stop_sent is not None:
            return 'stopped', '#777'
        return f'exited {code}', '#e04040'


class RemoteJob(Job):
    """A command run on the Pi through ssh."""

    host = HOST

    def __init__(self, tag, name, command, out):
        super().__init__(tag, name, command, out)
        self.argv = ['ssh', '-tt', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                     self.host, 'bash -ic ' + shlex.quote(command)]
        self.pkill_sent = None

    def start(self):
        if self.running():
            return
        self.say(f'{self.host} $ {self.command}')
        self.popen(subprocess.PIPE)
        self.pkill_sent = None

    def stop(self):
        if not self.running() or self.stop_sent is not None:
            return
        self.say('stopping (Ctrl+C on the Pi)')
        try:
            self.proc.stdin.write('\x03')
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        self.stop_sent = time.monotonic()

    def launch_file(self):
        """The word to pkill by: the .launch.py name, else the command."""
        return next((w for w in self.command.split() if w.endswith('.launch.py')),
                    self.command)

    def tick(self):
        if not self.running() or self.stop_sent is None:
            return
        now = time.monotonic()
        if self.pkill_sent is None and now - self.stop_sent > STOP_WAIT:
            name = self.launch_file()
            self.say(f'still running - pkill -INT -f {name} on the Pi')
            subprocess.Popen(
                ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', self.host,
                 'pkill -INT -f ' + shlex.quote(name)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.pkill_sent = now
        elif self.pkill_sent is not None and now - self.pkill_sent > STOP_WAIT:
            self.say('closing the ssh connection')
            self.signal(signal.SIGTERM)
            self.pkill_sent = now + 3600    # once


class SignBoard:
    """The ESP32 sign, over MQTT (paho 1.x). Callbacks only set attributes.

    paho runs in its own thread and reconnects by itself; the window reads
    `state()` on its tick, so nothing here touches Tk.
    """

    SIGN, ASK, ANSWER = '/control/sign', '/cmd/state', '/response/state'

    def __init__(self, host, port, out):
        self.host, self.port, self.out = host, port, out
        self.connected = False
        self.problem = 'connecting ...'     # why not connected, for the label
        self.since = time.monotonic()       # when connected / problem began
        self.answer = None                  # (word, monotonic)
        self.asked = None                   # monotonic of the last /cmd/state
        self.client = None
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.problem = 'paho-mqtt is not installed (pip install paho-mqtt)'
            self.say(self.problem)
            return
        c = self.client = mqtt.Client(client_id=f'arx-launcher-{os.getpid()}')
        c.on_connect = self.on_connect
        c.on_disconnect = self.on_disconnect
        c.on_message = self.on_message
        c.reconnect_delay_set(min_delay=1, max_delay=5)
        c.connect_async(host, port, keepalive=10)
        c.loop_start()
        threading.Thread(target=self.watch_first, daemon=True).start()

    def say(self, text):
        self.out.append(f'[sign] {text}')

    def where(self):
        return f'{self.host}:{self.port}'

    def watch_first(self):
        """Not connected after 5 s: find out why, in words."""
        time.sleep(5.0)
        if self.connected:
            return
        try:
            socket.create_connection((self.host, self.port), timeout=2.0).close()
            why = 'the port answers but MQTT did not connect'
        except ConnectionRefusedError:
            why = 'connection refused - nothing listening on that port'
        except socket.timeout:
            why = 'no answer - board off, or not on this network?'
        except OSError as e:
            why = f'{e.strerror or e}'
        self.problem = f'cannot connect to {self.where()} - {why}'
        self.since = time.monotonic()
        self.say(self.problem)

    def on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            self.connected = False
            self.problem = f'cannot connect to {self.where()} - broker refused (rc {rc})'
            self.since = time.monotonic()
            self.say(self.problem)
            return
        self.connected = True
        self.problem = None
        self.since = time.monotonic()
        client.subscribe(self.ANSWER)
        self.say(f'connected to {self.where()}')

    def on_disconnect(self, client, userdata, rc):
        if self.connected:
            self.say(f'lost the connection to {self.where()} (rc {rc}) - retrying')
        self.connected = False
        self.problem = f'lost connection to {self.where()} at {time.strftime("%H:%M:%S")} - retrying'
        self.since = time.monotonic()

    def on_message(self, client, userdata, msg):
        word = msg.payload.decode(errors='replace').strip().lower()
        self.answer = (word, time.monotonic())

    def send(self, word):
        """Publish to /control/sign. Returns whether it went; says so if not."""
        if self.client is None or not self.connected:
            self.say(f'NOT SENT: {word} - not connected to the board ({self.where()})')
            return False
        info = self.client.publish(self.SIGN, word)
        if info.rc != 0:
            self.say(f'NOT SENT: {word} - publish failed (rc {info.rc})')
            return False
        self.say(f'sent {self.SIGN} {word}')
        return True

    def ask(self):
        if self.client is not None and self.connected:
            self.asked = time.monotonic()
            self.client.publish(self.ASK, '?')

    def close(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()


class Window:

    def __init__(self, commands=COMMANDS, mqtt_host='192.168.0.102', mqtt_port=1883):
        import tkinter as tk
        from tkinter import ttk
        self.out = collections.deque()   # lines from the pump threads
        self.jobs = [(group, (RemoteJob if group == 'pi' else Job)(tag, name, cmd, self.out))
                     for group, tag, name, cmd in commands]
        self.calib = [j for g, j in self.jobs if g == 'calibration']
        self.jobs = [j for _, j in self.jobs]

        root = self.root = tk.Tk()
        root.title('arx launcher')
        root.protocol('WM_DELETE_WINDOW', self.close)

        self.rows = []

        def group(title, jobs):
            box = ttk.LabelFrame(root, text=title, padding=8)
            box.pack(fill='x', padx=8, pady=(8, 0))
            return box

        def row(box, i, job):
            b = tk.Button(box, text='start', width=7,
                          command=lambda j=job: self.toggle(j))
            b.grid(row=i, column=0, pady=2, sticky='w')
            ttk.Label(box, text=job.name, width=22).grid(row=i, column=1, padx=8, sticky='w')
            led = tk.Label(box, text='\u25cf', fg='#777', font=('TkDefaultFont', 14))
            led.grid(row=i, column=2, sticky='w')
            s = tk.Label(box, text='stopped', width=12, anchor='w', fg='#777')
            s.grid(row=i, column=3, sticky='w')
            ttk.Label(box, text=job.shown, foreground='#777'
                      ).grid(row=i, column=4, padx=8, sticky='w')
            self.rows.append((job, b, s, led))

        pi = [j for j in self.jobs if isinstance(j, RemoteJob)]
        if pi:
            box = group(f'Pi  ({RemoteJob.host})', pi)
            for i, job in enumerate(pi):
                row(box, i, job)

        self.all_button = None
        if self.calib:
            box = group('calibration  (this PC)', self.calib)
            self.all_button = tk.Button(box, text='1  start all', height=2,
                                        font=('TkDefaultFont', 12, 'bold'),
                                        command=self.toggle_calib)
            self.all_button.grid(row=0, column=0, columnspan=5, sticky='we', pady=(0, 6))
            for i, job in enumerate(self.calib):
                row(box, i + 1, job)
            root.bind('1', lambda e: self.toggle_calib())

        # The junction sign on the ESP32 - see SignBoard and the docstring.
        self.sign = SignBoard(mqtt_host, mqtt_port, self.out)
        self.sign_side = tk.StringVar(value='left')
        self.sign_delay = tk.StringVar(value='5')
        self.sign_active = False
        self.sign_due = None          # (monotonic, side) of the pending send
        self.sign_down_ok = True      # whether the last `down` went out
        self.sign_sent = ''           # what the countdown line last said
        self.sign_next_ask = time.monotonic() + 2.0
        box = group(f'sign board  (ESP32 MQTT {mqtt_host})', [])
        self.sign_led = tk.Label(box, text='\u25cf', fg='#d4ac0d', font=('TkDefaultFont', 14))
        self.sign_led.grid(row=0, column=0, sticky='w')
        ttk.Combobox(box, textvariable=self.sign_side, values=('left', 'right'),
                     state='readonly', width=6).grid(row=0, column=1, padx=4)
        tk.Label(box, text='after').grid(row=0, column=2)
        ttk.Combobox(box, textvariable=self.sign_delay,
                     values=('0', '1', '2', '3', '5', '8', '10', '15', '20', '30'),
                     state='readonly', width=4).grid(row=0, column=3, padx=4)
        tk.Label(box, text='s').grid(row=0, column=4, sticky='w')
        self.sign_button = tk.Button(box, width=12, font=('TkDefaultFont', 11, 'bold'),
                                     command=self.toggle_sign)
        self.sign_button.grid(row=0, column=5, padx=8)
        self.sign_link = tk.Label(box, text='', anchor='w')
        self.sign_link.grid(row=1, column=0, columnspan=8, sticky='w')
        self.sign_line = tk.Label(box, text='', anchor='w', fg='#555')
        self.sign_line.grid(row=2, column=0, columnspan=8, sticky='w')
        self.paint_sign()

        # The full stack, from a chosen preset. It starts its own camera
        # pipeline (intrinsic and extrinsic), so starting it stops the
        # calibration group first - two of each fight over the same topics.
        names = presets()
        self.preset = tk.StringVar(value='mission4' if 'mission4' in names else names[0])
        self.mission = Job('run', 'full.launch', MISSION_CMD.format(self.preset.get()), self.out)
        self.jobs.append(self.mission)
        box = group('mission  (this PC)', [self.mission])
        # The one button a run is started from, so it is the big one: green
        # to start, red to stop, the preset on it, and key 2 does the same.
        self.mission_button = tk.Button(
            box, height=2, width=26, font=('TkDefaultFont', 14, 'bold'),
            fg='white', activeforeground='white', command=self.toggle_mission)
        self.mission_button.grid(row=0, column=0, columnspan=5, sticky='we', pady=(0, 6))
        root.bind('2', lambda e: self.toggle_mission())
        tk.Label(box, text='preset').grid(row=1, column=0, sticky='w')
        ttk.Combobox(box, textvariable=self.preset, values=names, state='readonly',
                     width=12, font=('TkDefaultFont', 12, 'bold')
                     ).grid(row=1, column=1, padx=8, sticky='w')
        led = tk.Label(box, text='\u25cf', fg='#777', font=('TkDefaultFont', 14))
        led.grid(row=1, column=2, sticky='w')
        st = tk.Label(box, text='stopped', width=12, anchor='w', fg='#777')
        st.grid(row=1, column=3, sticky='w')
        self.mission_cmd = ttk.Label(box, text=self.mission.shown, foreground='#777')
        self.mission_cmd.grid(row=1, column=4, padx=8, sticky='w')
        self.preset.trace_add('write', lambda *a: self.pick_preset())
        self.rows.append((self.mission, None, st, led))
        self.mission_pending = False
        self.paint_mission_button()

        bar = ttk.Frame(root, padding=(8, 6, 8, 4))
        bar.pack(fill='x')
        ttk.Button(bar, text='Stop everything', command=self.stop_all).pack(side='left')
        ttk.Button(bar, text='Clear output',
                   command=lambda: self.log.delete('1.0', 'end')).pack(side='left', padx=6)

        frame = ttk.Frame(root, padding=(8, 0, 8, 8))
        frame.pack(fill='both', expand=True)
        self.log = tk.Text(frame, height=18, width=110, bg='#111', fg='#ddd',
                           insertbackground='#ddd', font=('TkFixedFont', 9), wrap='none')
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        root.after(100, self.tick)

    def paint_mission_button(self):
        """Green START with the preset on it, or red STOP while it runs."""
        if self.mission.running():
            self.mission_button.config(
                text=f'\u25a0  2  STOP  {self.preset.get()}',
                bg='#c0392b', activebackground='#e74c3c')
        elif self.mission_pending:
            self.mission_button.config(
                text=f'\u2026  stopping calibration first',
                bg='#b7950b', activebackground='#d4ac0d')
        else:
            self.mission_button.config(
                text=f'\u25b6  2  START  {self.preset.get()}',
                bg='#1e8449', activebackground='#27ae60')

    def pick_preset(self):
        """Point the mission job at the chosen preset - not while it runs."""
        if self.mission.running():
            return
        cmd = MISSION_CMD.format(self.preset.get())
        self.mission.command = self.mission.shown = cmd
        self.mission.argv = shlex.split(cmd)
        self.mission_cmd.config(text=cmd)

    def toggle_mission(self):
        if self.mission.running():
            self.mission.stop()
            return
        busy = [j for j in self.calib if j.running()]
        for job in busy:
            job.stop()
        self.pick_preset()
        if busy:
            # Started from tick() once they are down, not alongside them.
            self.out.append('[run] stopping the calibration group first - '
                            'full.launch starts its own camera pipeline')
            self.mission_pending = True
            return
        self.start_mission()

    def start_mission(self):
        """full.launch, and with the sign board active: down now, the side later."""
        self.mission.start()
        if self.sign_active:
            self.sign_down_ok = self.sign.send('down')
            delay = float(self.sign_delay.get())
            self.sign_due = (time.monotonic() + delay, self.sign_side.get())
            self.sign_sent = ''

    def toggle_sign(self):
        self.sign_active = not self.sign_active
        if self.sign_active:
            self.sign_down_ok = self.sign.send('down')
        self.sign_due = None
        self.sign_sent = ''
        self.paint_sign()

    def paint_sign(self):
        """LED, connection line, button and countdown - from tick."""
        b = self.sign
        if b.connected:
            self.sign_led.config(fg='#1e8449')
            if b.answer is None:
                ans = 'board: no answer yet'
            else:
                ans = f'board: {b.answer[0]}  ({time.monotonic() - b.answer[1]:.0f} s ago)'
            if (b.asked is not None and time.monotonic() - b.asked > 3.0
                    and (b.answer is None or b.answer[1] < b.asked)):
                ans += '   - board did not answer /cmd/state'
            self.sign_link.config(text=f'connected to {b.where()}   {ans}', fg='#1e8449')
        else:
            starting = b.problem == 'connecting ...'
            self.sign_led.config(fg='#d4ac0d' if starting else '#c0392b')
            self.sign_link.config(text=b.problem, fg='#9a7d0a' if starting else '#c0392b')
        if self.sign_active:
            self.sign_button.config(text='\u25cf ACTIVE', bg='#1e8449', fg='white',
                                    activebackground='#27ae60', activeforeground='white')
        else:
            self.sign_button.config(text='active', bg='#d5d8dc', fg='black',
                                    activebackground='#e5e7e9', activeforeground='black')
        if self.sign_due is not None:
            left = self.sign_due[0] - time.monotonic()
            down = '' if self.sign_down_ok else 'down NOT SENT (not connected) - '
            self.sign_line.config(text=f'{down}sending {self.sign_due[1]} in {max(left, 0):.0f} s',
                                  fg='#c0392b' if down else '#555')
        else:
            self.sign_line.config(text=self.sign_sent,
                                  fg='#c0392b' if self.sign_sent.startswith('NOT') else '#555')

    def tick_sign(self):
        now = time.monotonic()
        if self.sign_due is not None:
            if not self.mission.running():
                self.sign_due = None
                self.sign_sent = 'full.launch stopped before the delay - nothing sent'
            elif now >= self.sign_due[0]:
                side = self.sign_due[1]
                self.sign_due = None
                self.sign_sent = (f'sent {side} at {time.strftime("%H:%M:%S")}'
                                  if self.sign.send(side)
                                  else f'NOT SENT: {side} - not connected to the board')
        if now >= self.sign_next_ask:
            self.sign_next_ask = now + 10.0
            self.sign.ask()
        self.paint_sign()

    def toggle(self, job):
        if job.running():
            job.stop()
        else:
            job.start()

    def toggle_calib(self):
        if any(j.running() for j in self.calib):
            for job in self.calib:
                job.stop()
        else:
            for job in self.calib:
                job.start()

    def stop_all(self):
        for job in self.jobs:
            job.stop()

    def tick(self):
        if self.mission_pending and not any(j.running() for j in self.calib):
            self.mission_pending = False
            self.start_mission()
        self.tick_sign()
        if self.out:
            at_end = self.log.yview()[1] > 0.999
            while self.out:
                self.log.insert('end', self.out.popleft() + '\n')
            extra = int(self.log.index('end-1c').split('.')[0]) - LOG_LINES
            if extra > 0:
                self.log.delete('1.0', f'{extra + 1}.0')
            if at_end:
                self.log.see('end')
        for job, b, s, led in self.rows:
            job.tick()
            text, colour = job.status()
            s.config(text=text, fg=colour)
            led.config(fg=colour)
            if b is not None:
                b.config(text='stop' if job.running() else 'start')
        self.paint_mission_button()
        if self.all_button is not None:
            self.all_button.config(
                text='1  stop all' if any(j.running() for j in self.calib)
                else '1  start all')
        self.root.after(200, self.tick)

    def close(self):
        """Stop everything, wait for it, then go."""
        self.stop_all()
        # Twice the wait, ticking, so a Pi job gets to try Ctrl+C and then
        # pkill before the connection is cut below.
        deadline = time.monotonic() + 2 * STOP_WAIT + 1
        while any(j.running() for j in self.jobs) and time.monotonic() < deadline:
            for job in self.jobs:
                job.tick()
            time.sleep(0.1)
        for job in self.jobs:
            if job.running():
                job.signal(signal.SIGKILL)
        self.sign.close()
        self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default=HOST, help=f'ssh target for the Pi (default {HOST})')
    ap.add_argument('--mqtt-host', default='192.168.0.102',
                    help='the sign board\'s MQTT broker (default 192.168.0.102)')
    ap.add_argument('--mqtt-port', type=int, default=1883)
    args = ap.parse_args()
    RemoteJob.host = args.host
    Window(mqtt_host=args.mqtt_host, mqtt_port=args.mqtt_port).run()


if __name__ == '__main__':
    main()

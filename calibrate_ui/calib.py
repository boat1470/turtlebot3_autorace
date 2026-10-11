#!/usr/bin/env python3
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""Save what a detector is running with back into its param file.

    python3 calibrate_ui/calib.py              a window, with a Save button
    python3 calibrate_ui/calib.py --list       what it knows how to save
    python3 calibrate_ui/calib.py light        save that one and exit

Tuning a detector means dragging sliders in rqt_reconfigure and watching an
image. rqt_reconfigure has no way to write what you arrived at back to disk,
so the values live only in the running node: close the terminal and an hour
of work is gone. Everything in param/ was until now copied across by hand,
one number at a time, which is slow and gets a digit wrong eventually.

This does not replace rqt_reconfigure. It has no sliders - the ones there are
already correct, because every threshold declares an IntegerRange and the
slider picks its travel up from that. This is the Save button rqt_reconfigure
is missing, and it is meant to sit beside it.

Two things make it simple enough to be worth having.

The first is that a parameter name is already a path into the yaml:

    light   red.hue_l                ->  red:    hue_l:
    lane    detect.lane.white.hue_l  ->  detect:  lane:  white:  hue_l:
    level   level.red.hue_lo_l       ->  level:  red:   hue_lo_l:

so one walker serves every detector, and new ones need a line in TARGETS and
nothing else.

The second is that only the number on the leaf line is rewritten. The files in
param/ are mostly comments - where a threshold came from, what it was measured
against, what breaks if it moves - and those are the part worth keeping. A
yaml library would reformat the file and drop them, which is why this does not
use one. ruamel.yaml would round-trip them, and is not installed.

The node is found rather than named. A detector answers to one name under
`ros2 launch` and another under `ros2 run` - arx_light_detector against
detect_traffic_light - and asking which parameters a node has is a surer test
than guessing which way it was started.
"""

import argparse
import os
import re
import sys
import time

# target -> (prefixes to save, param file relative to the repo root)
#
# The prefixes are parameter namespaces, not node names. Anything outside them
# is left alone: frame_skip, publish_debug_image and always_on are run-time
# choices, not calibration, and writing them back would quietly turn debug
# images on for a race.
TARGETS = {
    'light': (('red', 'yellow', 'green', 'roi'), 'arx_mission/param/light.yaml'),
    'lane': (('detect.lane',), 'arx_mission/param/lane.yaml'),
    'level': (('level',), 'arx_mission/param/level.yaml'),
    'sign': (('sign',), 'arx_mission/param/sign.yaml'),
    'obstacle': (('obstacle',), 'arx_mission/param/obstacle.yaml'),
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wanted(name, prefixes):
    """Whether a parameter belongs to one of the namespaces being saved."""
    return any(name == p or name.startswith(p + '.') for p in prefixes)


def fetch(prefixes, timeout=5.0):
    """Ask every live node for its parameters and return those that match.

    Returns (node_name, {param: value}). The first node holding any matching
    parameter wins; there is only ever one detector of a kind running, and if
    there are two the right answer is to stop one rather than to choose here.
    """
    import rclpy
    from rclpy.node import Node
    from rcl_interfaces.srv import GetParameters, ListParameters

    own = rclpy.ok()
    if not own:
        rclpy.init()
    probe = Node('arx_calib_probe')

    def call(srv, name, node, req, wait):
        # The service name spelled out. srv.__name__.lower() gives
        # 'listparameters', which is not what a node advertises, and every
        # call then waits out its timeout against a service that is not there.
        cli = probe.create_client(srv, f'{node}/{name}')
        try:
            if not cli.wait_for_service(timeout_sec=wait):
                return None
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(probe, fut, timeout_sec=timeout)
            return fut.result()
        finally:
            probe.destroy_client(cli)

    try:
        # Spin before asking. A node that has only just been created has an
        # empty graph: discovery is asynchronous, and the first call returns
        # whatever has arrived so far, which on a quiet moment is nothing.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.1)
        # Namespace included. /camera/image_projection answers on
        # /camera/image_projection/list_parameters, and the name alone would
        # look for a service that is not there and time out on every detector
        # that a launch file put under a namespace.
        names = [(ns.rstrip('/') + '/' + n)
                 for n, ns in probe.get_node_names_and_namespaces()]
        for node in sorted(names):
            if node.endswith('/' + probe.get_name()):
                continue
            # Short wait while looking: most of these nodes are not the one,
            # and a composed node does not offer parameter services at all.
            listed = call(ListParameters, 'list_parameters', node,
                          ListParameters.Request(), 0.5)
            if listed is None:
                continue
            keys = sorted(k for k in listed.result.names if wanted(k, prefixes))
            if not keys:
                continue
            req = GetParameters.Request()
            req.names = keys
            got = call(GetParameters, 'get_parameters', node, req, timeout)
            if got is None:
                continue
            return node, {k: _plain(v) for k, v in zip(keys, got.values)}
        return None, {}
    finally:
        probe.destroy_node()
        if not own:
            rclpy.shutdown()


def _plain(v):
    """Turn a ParameterValue into something that can be written to yaml."""
    t = v.type
    return {1: v.bool_value, 2: v.integer_value, 3: v.double_value,
            4: v.string_value}.get(t)


def _block(lines, start, indent):
    """Return the half-open line range belonging to a key opened at `start`."""
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith('#'):
            if len(line) - len(line.lstrip()) <= indent:
                break
        i += 1
    return start + 1, i


def find_leaf(lines, dotted):
    """Return the line index holding `dotted`, or None.

    Walks the file the way the name reads, one key at a time, so a key that
    appears under two parents cannot be confused for the other - `red` is a
    colour in light.yaml and a namespace in level.yaml.
    """
    lo, hi = 0, len(lines)
    # Every param file nests its keys under ros__parameters.
    for i in range(lo, hi):
        if lines[i].strip() == 'ros__parameters:':
            lo, hi = _block(lines, i, len(lines[i]) - len(lines[i].lstrip()))
            break
    parts = dotted.split('.')
    for depth, part in enumerate(parts):
        last = depth == len(parts) - 1
        pat = re.compile(rf'^(\s*){re.escape(part)}:\s*(.*)$')
        for i in range(lo, hi):
            m = pat.match(lines[i])
            if not m:
                continue
            if last:
                return i if m.group(2).strip() else None
            lo, hi = _block(lines, i, len(m.group(1)))
            break
        else:
            return None
    return None


def render(value, old):
    """`value` formatted the way `old` was, so the diff stays to the digits."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        # Keep a trailing .0 rather than turning a float key into an int one.
        return repr(value) if '.' in old or 'e' in old.lower() else f'{value:.1f}'
    return str(value)


def apply(path, values):
    """Write `values` into `path`. Returns (changed, missing)."""
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    changed, missing = [], []
    for name in sorted(values):
        i = find_leaf(lines, name)
        if i is None:
            missing.append(name)
            continue
        key, _, rest = lines[i].partition(':')
        old = rest.split('#')[0].strip()
        new = render(values[name], old)
        if old == new:
            continue
        lines[i] = lines[i].replace(f':{rest}', f': {new}' +
                                    (' #' + rest.split('#', 1)[1] if '#' in rest else ''), 1)
        changed.append((name, old, new))
    if changed:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))
    return changed, missing


def save(target, quiet=False):
    prefixes, rel = TARGETS[target]
    node, values = fetch(prefixes)
    if not values:
        return None, f'no running node has {"/".join(prefixes)} parameters'
    path = os.path.join(REPO, rel)
    changed, missing = apply(path, values)
    if not quiet:
        for name, old, new in changed:
            print(f'  {name:38s} {old:>8s} -> {new}')
        for name in missing:
            print(f'  {name:38s} not in {rel}', file=sys.stderr)
    return node, (f'{len(changed)} changed in {rel}' +
                  (f', {len(missing)} not found' if missing else ''))


def gui():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title('arx calib')
    pick = tk.StringVar(value='light')
    status = tk.StringVar(value='pick a detector, then Read')

    top = ttk.Frame(root, padding=8)
    top.pack(fill='x')
    ttk.Label(top, text='detector').pack(side='left')
    ttk.Combobox(top, textvariable=pick, values=sorted(TARGETS),
                 width=12, state='readonly').pack(side='left', padx=6)

    cols = ('param', 'file', 'node')
    table = ttk.Treeview(root, columns=cols, show='headings', height=18)
    for c, w in zip(cols, (300, 90, 90)):
        table.heading(c, text=c)
        table.column(c, width=w, anchor='w' if c == 'param' else 'e')
    table.pack(fill='both', expand=True, padx=8)
    table.tag_configure('diff', background='#fff2cc')

    def read():
        table.delete(*table.get_children())
        prefixes, rel = TARGETS[pick.get()]
        node, values = fetch(prefixes)
        if not values:
            status.set(f'no running node has {"/".join(prefixes)} parameters')
            return
        path = os.path.join(REPO, rel)
        with open(path, encoding='utf-8') as fh:
            lines = fh.read().split('\n')
        n_diff = 0
        for name in sorted(values):
            i = find_leaf(lines, name)
            old = '-' if i is None else lines[i].partition(':')[2].split('#')[0].strip()
            new = render(values[name], old)
            tag = ('diff',) if old != new else ()
            n_diff += bool(tag)
            table.insert('', 'end', values=(name, old, new), tags=tag)
        status.set(f'{node}: {n_diff} differ from {rel}')

    def write():
        node, msg = save(pick.get(), quiet=True)
        status.set(msg if node else msg)
        read()

    bar = ttk.Frame(root, padding=8)
    bar.pack(fill='x')
    ttk.Button(bar, text='Read', width=10, command=read).pack(side='left')
    ttk.Button(bar, text='Save', width=10, command=write).pack(side='left', padx=6)
    ttk.Label(root, textvariable=status, padding=(8, 0, 8, 8),
              foreground='#555').pack(fill='x')
    read()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('target', nargs='?', choices=sorted(TARGETS))
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    if args.list:
        for k, (p, rel) in sorted(TARGETS.items()):
            print(f'  {k:10s} {", ".join(p):26s} {rel}')
        return 0
    if args.target:
        node, msg = save(args.target)
        print(f'{node or "-"}: {msg}')
        return 0 if node else 1
    gui()
    return 0


if __name__ == '__main__':
    sys.exit(main())

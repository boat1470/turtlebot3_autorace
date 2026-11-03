# Fit odom onto curve_map: piecewise rigid corrections, pulled so that while
# detect_lane sees both lines (lane_state 2) the robot sits half a lane from
# the nearest line.
import sys, math, numpy as np
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'track.py')).read().split("img=mpimg")[0])
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
out, path = sys.argv[1], sys.argv[2]
SEG_S = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
d = load(path); o = d['/odom']
first = [c[0] for c in d['/cmd_vel'] if abs(c[1]) > 0.01][0]
img = mpimg.imread('/home/boat1470/turtlebot3_ws/src/turtlebot3_autorace/docs/curve_map.png')[:, :, :3]
H = img.shape[0]; MM = 4000.0 / H
r, g, b = img[..., 0], img[..., 1], img[..., 2]
lines = ((r > 0.9) & (g > 0.9) & (b > 0.9)) | ((r > 0.85) & (g > 0.75) & (b < 0.3))
dist = distance_transform_edt(~lines) * MM          # mm to nearest line pixel
def dist_at(x, y):                                   # world mm -> mm
    return map_coordinates(dist, [H - 1 - y / MM, x / MM], order=1, mode='nearest')

t = np.array([p[0] for p in o]); P = np.array([world(p[1], p[2]) for p in o])   # SCALE 1.00
ls = d['/detect/lane_state']; lt = np.array([s for s, v in ls]); lv = np.array([v for s, v in ls])
state = lv[np.clip(np.searchsorted(lt, t) - 1, 0, len(lv) - 1)]
use = (state == 2) & (t >= first)
seg = np.floor((t - first) / SEG_S).astype(int); seg[t < first] = 0
nseg = seg.max() + 1
# Rebuild the path from odom's own steps, each turned by a slowly varying
# heading correction and stretched by a per-segment scale (wheel slip), from
# a fitted start pose. Continuous by construction.
step = np.diff(P0 := np.array([[p[1], p[2]] for p in o]) * 1000.0, axis=0)
step = np.c_[step[:, 1], -step[:, 0]]           # odom (x fwd down, y left) -> world mm, as world()
HALF = 125.0
def apply(x):
    x0, y0, th0, sg = x[0], x[1], x[2], x[3]
    ph = np.cumsum(x[4:4+nseg]); sc = x[4+nseg:4+2*nseg]
    k = seg[1:]
    a = th0 + ph[k]; c, s_ = np.cos(a), np.sin(a)
    f = (1.0 + sg + sc[k])[:, None]
    v = np.c_[c*step[:, 0] - s_*step[:, 1], s_*step[:, 0] + c*step[:, 1]] * f
    q = np.vstack([[0, 0], np.cumsum(v, axis=0)]) + [245 + x0, 1188 + y0]
    return q
def res(x):
    q = apply(x)
    r1 = (dist_at(q[use, 0], q[use, 1]) - HALF) / 30.0
    r2 = x[4:4+nseg] / 0.01                     # heading correction changes slowly
    r3 = x[4+nseg:] / 0.02                      # per-segment slip small
    r4 = np.r_[x[0]/30, x[1]/30, x[2]/0.03, x[3]/0.05]
    return np.r_[r1, r2, r3, r4]
DEG = math.radians(1)
lo = np.r_[-60, -60, -3*DEG, -0.04, [-2*DEG]*nseg, [-0.06]*nseg]
x = least_squares(res, np.zeros(4+2*nseg), bounds=(lo, -lo), loss='soft_l1', f_scale=1.0, max_nfev=400).x
Q = apply(x); P = apply(np.zeros_like(x))
before = np.median(np.abs(dist_at(P[use, 0], P[use, 1]) - HALF))
after = np.median(np.abs(dist_at(Q[use, 0], Q[use, 1]) - HALF))
print(f'segments {nseg} x {SEG_S}s   median |dist-{HALF:.0f}| on lane_state 2: {before:.0f} -> {after:.0f} mm')
print(f'start shift {x[0]:+.0f},{x[1]:+.0f} mm, start yaw {np.degrees(x[2]):+.1f} deg, scale {1+x[3]:.3f}, heading corr {np.degrees(np.cumsum(x[4:4+nseg]).min()):+.1f}..{np.degrees(np.cumsum(x[4:4+nseg]).max()):+.1f} deg, slip {x[4+nseg:].min():+.3f}..{x[4+nseg:].max():+.3f}')

fig, ax = plt.subplots(figsize=(13, 13)); ax.imshow(mpimg.imread('/home/boat1470/turtlebot3_ws/src/turtlebot3_autorace/docs/curve_map.png'), extent=[0, 4000, 0, 4000])
ax.plot(P[t >= first, 0], P[t >= first, 1], color='grey', lw=1, alpha=0.6, label='odom as recorded')
stages = [(s, m.split('-> ')[1].split(':')[0]) for s, m in d['/rosout'] if m.startswith('stage ') and '->' in m]
cols = plt.cm.tab20.colors; names = []
for i, (s, n) in enumerate(stages):
    e = stages[i+1][0] if i + 1 < len(stages) else 1e18
    m = (t >= max(s, first)) & (t <= e)
    if m.sum() < 2: continue
    lab = None if n in names else n
    if n not in names: names.append(n)
    ax.plot(Q[m, 0], Q[m, 1], color=cols[names.index(n) % 20], lw=2.4, label=lab)
keys = [('white line gone', 'arc'), ('on the spot -', 'spin'), ('park phase 1', 'back in'), ('park phase 2', 'parked'), ('bar down', 'bar'), ('tunnel front', 'tunnel')]
for s, msg in d['/rosout']:
    for k, l in keys:
        if k in msg:
            i = int(np.argmin(np.abs(t - s)))
            ax.plot(Q[i, 0], Q[i, 1], 'o', color='white', ms=6, mec='k')
            ax.annotate(f'{l} {s-first:.1f}s', Q[i], xytext=(6, 6), textcoords='offset points', color='white', fontsize=9, bbox=dict(fc='black', alpha=0.6, lw=0))
ax.set_xlim(0, 4000); ax.set_ylim(0, 4000); ax.legend(loc='upper left', fontsize=9)
ax.set_title(f'{path.rstrip("/").split("/")[-1]} - odom fitted to the lane (grey: as recorded)')
plt.savefig(out, dpi=70, bbox_inches='tight')

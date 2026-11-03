# 18B close-up: how the steering oscillates.  python3 zoom18b.py <bag> <out_dir>
import os, sys, numpy as np, cv2
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'track.py')).read().split("img=mpimg")[0])
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
path, outd = sys.argv[1], sys.argv[2]
d = load(path)
first = [c[0] for c in d['/cmd_vel'] if abs(c[1]) > 0.01][0]
a = [s for s, m in d['/rosout'] if 'park phase 5' in m][0]; b = [s for s, m in d['/rosout'] if 'parking done' in m][0]
A, B = a - 0.3, b + 0.6
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap' if path.endswith('.mcap') else ''), rosbag2_py.ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = [t for t in ('/detect/lane_ahead', '/detect/image_lane/compressed') if t in types]
r.set_filter(rosbag2_py.StorageFilter(topics=want))
ahead, frames = [], []
while r.has_next():
    t, raw, ts = r.read_next(); s = ts * 1e-9
    if not (A <= s <= B): continue
    m = deserialize_message(raw, get_message(types[t]))
    if t == '/detect/lane_ahead': ahead.append((s, m.data))
    else: frames.append((s, cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)))
T = lambda s: s - first
L = [(s, v) for s, v in d['/detect/lane'] if A <= s <= B]
lt = np.array([s for s, v in L]); lv = np.array([v for s, v in L])
e = lv - 500; dt = np.diff(lt); de = np.r_[0, np.diff(e) / np.where(dt > 1e-3, dt, np.nan)]
C = [c for c in d['/cmd_vel'] if A <= c[0] <= B]
cv_at = lambda s: [c for c in d['/cmd_vel'] if c[0] <= s][-1]
kp, ref = 0.004, 0.07
kd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0023   # control.kd the run used
Pt = np.array([-kp * max(cv_at(s)[1], 0.04) / ref * x for s, x in zip(lt, e)])
Dt = -kd * np.nan_to_num(de)
o = [p for p in d['/odom'] if A <= p[0] <= B]

BLUE, ORANGE, AQUA, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GRID,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2})
fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2.4, 1.4, 1.4]})
for x in ax: x.grid(axis='y', color=GRID, lw=0.6)
x = ax[0]
x.plot(T(lt), lv, color=BLUE, lw=1.8, marker='o', ms=3, label='lane centre, row 580 (steers)')
x.plot([T(s) for s, v in ahead], [v for s, v in ahead], color=ORANGE, lw=1.4, label='lane centre, row 300 (only slows)')
x.axhline(500, color=INK2, lw=0.8, ls='--'); x.text(T(a) + 3.6, 500, ' robot centre 500', va='center', fontsize=8, color=INK2)
x.set_ylabel('px in projected image'); x.legend(loc='lower left', frameon=False)
x = ax[1]
x.step([T(c[0]) for c in C], [c[2] for c in C], where='post', color=INK2, lw=2.2, label='ω sent = P + D')
x.plot(T(lt), Pt, color=BLUE, lw=1.4, label='P part (kp × error, scaled by speed)')
x.plot(T(lt), Dt, color=ORANGE, lw=1.4, label=f'D part (kd {kd} × error rate)')
x.axhline(0, color=INK2, lw=0.6); x.set_ylabel('turn rate  rad/s'); x.legend(loc='lower right', frameon=False, ncol=3)
x = ax[2]
ot = np.array([T(p[0]) for p in o]); oy = np.unwrap(np.radians([p[3] for p in o]))
x.plot(ot, np.degrees(oy), color=BLUE, lw=1.8); x.set_ylabel('heading  deg')
x = ax[3]
x.step([T(c[0]) for c in C], [c[1] for c in C], where='post', color=BLUE, lw=1.8); x.set_ylabel('v sent  m/s')
x.set_xlabel('seconds from the first drive command')
# mark the frames shown below
cmdw = np.array([c[2] for c in C]); cmt = np.array([c[0] for c in C])
seg = (cmt > a + 1.3) & (cmt < a + 3.2)
idx = np.where(seg)[0]
pk = [i for i in idx[1:-1] if cmdw[i] >= cmdw[i-1] and cmdw[i] >= cmdw[i+1] and cmdw[i] > 0.6]
tr = [i for i in idx[1:-1] if cmdw[i] <= cmdw[i-1] and cmdw[i] <= cmdw[i+1] and cmdw[i] < 0.15]
def thin(ix):
    out = []
    for i in ix:
        if not out or cmt[i] - cmt[out[-1]] > 0.2: out.append(i)
    return out
picks = sorted([(cmt[i], 'turn hard') for i in thin(pk)][:3] + [(cmt[i], 'turn off') for i in thin(tr)][:3])
for k, (s, lab) in enumerate(picks):
    for xx in ax[:2]: xx.axvline(T(s), color=AQUA if lab == 'turn off' else ORANGE, lw=1, ls=':')
    ax[1].text(T(s), 1.12, str(k + 1), ha='center', fontsize=9, color=INK2)
Z0, Z1 = T(a) + 1.0, T(a) + 3.6
ax[0].set_xlim(Z0, Z1); ax[0].set_ylim(300, 540); ax[1].set_ylim(-0.6, 1.2); ax[2].set_ylim(85, 145)
fig.suptitle(f'18B leaving the lot - {path.rstrip("/").split("/")[-1]}', x=0.02, ha='left', fontsize=13)
plt.subplots_adjust(hspace=0.1, top=0.95)
plt.savefig(os.path.join(outd, '18B_oscillation.png'), dpi=80, bbox_inches='tight')
# frames strip
if frames:
    tiles = []
    for k, (s, lab) in enumerate(picks):
        f = min(frames, key=lambda q: abs(q[0] - s))[1].copy()
        h, w = f.shape[:2]; sc = 360 / h; f = cv2.resize(f, (int(w * sc), 360))
        c = cv_at(s)
        cv2.rectangle(f, (0, 0), (f.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(f, f'{k+1}  {T(s):.2f}s  {lab}  w {c[2]:+.2f}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        tiles.append(f)
    cv2.imwrite(os.path.join(outd, '18B_frames.png'), np.hstack(tiles))
print('picks', [(round(T(s), 2), l) for s, l in picks], 'frames', len(frames))

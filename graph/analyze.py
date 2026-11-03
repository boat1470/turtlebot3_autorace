import sys, math, numpy as np
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'track.py')).read().split("img=mpimg")[0])
out, path = sys.argv[1], sys.argv[2]
d = load(path); o = d['/odom']
first = [c[0] for c in d['/cmd_vel'] if abs(c[1]) > 0.01][0]
T = lambda s: s - first
ot = np.array([T(p[0]) for p in o]); ox = np.array([p[1] for p in o]); oy = np.array([p[2] for p in o]); yaw = np.array([p[3] for p in o])
# odom speed and yaw rate, smoothed over ~0.2 s
dt = np.gradient(ot); dt[dt <= 0] = 1e-3
v = np.hypot(np.gradient(ox), np.gradient(oy)) / dt
wy = np.gradient(np.unwrap(np.radians(yaw))) / dt
k = max(1, int(0.2 / np.median(dt)))
sm = lambda a: np.convolve(a, np.ones(k) / k, mode='same')
v, wy = sm(v), sm(wy)
ct = np.array([T(c[0]) for c in d['/cmd_vel']]); cv = np.array([c[1] for c in d['/cmd_vel']]); cw = np.array([c[2] for c in d['/cmd_vel']])
lt = np.array([T(s) for s, x in d['/detect/lane']]); le = np.array([x - 500 for s, x in d['/detect/lane']])
st = np.array([T(s) for s, x in d['/detect/lane_state']]); sv = np.array([x for s, x in d['/detect/lane_state']])
stages = [(T(s), m.split('-> ')[1].split(':')[0]) for s, m in d['/rosout'] if m.startswith('stage ') and '->' in m]
end = T([s for s, m in d['/rosout'] if 'tunnel front' in m][0]) if any('tunnel front' in m for s, m in d['/rosout']) else ot[-1]
events = []
for s, m in d['/rosout']:
    for key, lab in [('white line gone', 'arc start'), ('arc done', 'arc end'), ('on the spot -', 'spin end'),
                     ('park phase 1', 'back in'), ('park phase 2', 'parked'), ('park phase 5', 'out on yellow'),
                     ('bar down', 'bar down'), ('bar lifted', 'bar up'), ('tunnel front', 'tunnel')]:
        if key in m: events.append((T(s), lab))

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True, gridspec_kw={'height_ratios': [2.2, 2.2, 1.8, 2.2, 0.7]})
X0, X1 = -1, end + 1
for a in ax:
    a.grid(axis='y', color=GRID, lw=0.6); a.set_xlim(X0, X1)
    for i, (s, n) in enumerate(stages):
        e = stages[i + 1][0] if i + 1 < len(stages) else X1
        if i % 2: a.axvspan(s, e, color='#f0efec', lw=0, zorder=0)
    for s, l in events: a.axvline(s, color='#b5b3ab', lw=0.7, ls=':', zorder=1)
# stage names on top
for i, (s, n) in enumerate(stages):
    e = stages[i + 1][0] if i + 1 < len(stages) else X1
    if e - s > 1.2 and s < X1:
        ax[0].text((max(s, X0) + min(e, X1)) / 2, 1.02, n.replace('_', ' '), transform=ax[0].get_xaxis_transform(), ha='center', va='bottom', fontsize=7.5, color=INK2, rotation=0 if e - s > 6 else 40)
for s, l in events:
    ax[4].text(s, -0.9, l, rotation=90, fontsize=7, color=INK2, ha='center', va='top')

a = ax[0]; a.step(ct, cv, where='post', color=BLUE, lw=1.4, label='commanded v'); a.plot(ot, v, color=ORANGE, lw=1.4, label='odom v (measured)')
a.axhline(0.21, color='#d03b3b', lw=0.8, ls='--'); a.text(X1, 0.21, ' wheel cap 0.21', color='#d03b3b', fontsize=7.5, va='center')
a.set_ylabel('speed  m/s'); a.set_ylim(-0.25, 0.32); a.legend(loc='upper left', ncol=2, frameon=False)
a = ax[1]; a.step(ct, cw, where='post', color=BLUE, lw=1.2, label='commanded ω'); a.plot(ot, wy, color=ORANGE, lw=1.2, label='IMU yaw rate (measured)')
a.set_ylabel('turn rate  rad/s'); a.set_ylim(-3.6, 3.6); a.legend(loc='upper left', ncol=2, frameon=False)
yb = yaw.astype(float).copy(); yb[1:][np.abs(np.diff(yaw)) > 180] = np.nan
a = ax[2]; a.plot(ot, yb, color=BLUE, lw=1.4); a.set_ylabel('heading  deg'); a.set_yticks([-180, -90, 0, 90, 180]); a.set_ylim(-195, 195)
a = ax[3]; a.plot(lt, le, color=BLUE, lw=0.9); a.axhline(0, color=INK2, lw=0.6)
a.set_ylabel('lane error  px\n(centre − 500)'); a.set_ylim(-260, 260)
a = ax[4]
cols = {0: '#9c9a92', 1: YELLOW, 2: AQUA, 3: BLUE}; names = {0: 'none', 1: 'yellow only', 2: 'both', 3: 'white only'}
for i in range(len(st) - 1):
    if st[i] < X0: continue
    a.axvspan(st[i], st[i + 1], 0.1, 0.9, color=cols.get(int(sv[i]), '#ccc'), lw=0)
a.set_yticks([]); a.set_ylabel('lane_state', rotation=0, ha='right', va='center')
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=cols[k], label=names[k]) for k in (2, 1, 3, 0)], loc='upper center', bbox_to_anchor=(0.5, -1.6), ncol=4, frameon=False)
ax[-1].set_xlabel('seconds from the first drive command', labelpad=60)
fig.suptitle(f'{path.rstrip("/").split("/")[-1]} - tunnel front at {end:.1f} s', x=0.02, ha='left', fontsize=13, color=INK)
plt.subplots_adjust(hspace=0.12, top=0.93)
plt.savefig(out, dpi=80, bbox_inches='tight')

# numbers
print(f'end {end:.1f}')
for i, (s, n) in enumerate(stages):
    e = stages[i + 1][0] if i + 1 < len(stages) else end
    if n in ('stopped',): continue
    m = (ot >= s) & (ot < e); mc = (ct >= s) & (ct < e)
    if m.sum() < 3: continue
    vc = cv[mc]; vc = vc[vc > 0.01]
    print(f'{n:22s} {s:6.1f}-{e:6.1f} {e-s:5.1f}s  v odom {np.mean(v[m]):.3f}  cmd {np.mean(vc) if len(vc) else 0:.3f}  ratio {np.mean(v[m])/np.mean(vc) if len(vc) else 0:.2f}  max|w|cmd {np.max(np.abs(cw[mc])) if mc.any() else 0:.2f}')
for s, l in events: print(f'  {s:6.1f} {l}')

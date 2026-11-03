# Speed of two runs side by side, by distance along the course.
#   python3 speedcmp.py out.png <base run> <run> [<run> ...]   (time behind is against the first)
import os, sys, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'track.py')).read().split("img=mpimg")[0])
out, runs = sys.argv[1], sys.argv[2:]
def prep(run):
    d = load(os.path.expanduser('~/runs/' + run)); o = d['/odom']
    first = [c[0] for c in d['/cmd_vel'] if abs(c[1]) > 0.01][0]
    end = [s for s, m in d['/rosout'] if 'tunnel front' in m][0]
    t = np.array([p[0] for p in o]); x = np.array([p[1] for p in o]); y = np.array([p[2] for p in o])
    # clock from the wheels first turning: /odom can reach the bag a second late
    first = t[np.argmax(np.hypot(x - x[0], y - y[0]) > 0.005)]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    keep = (t >= first - 0.5) & (t <= end + 0.3)
    t, s = t[keep], s[keep] - s[keep][0]
    dt = np.gradient(t); dt[dt <= 0] = 1e-3
    v = np.gradient(s) / dt; k = 6; v = np.convolve(v, np.ones(k) / k, 'same')
    c = [(cc[0], cc[1]) for cc in d['/cmd_vel'] if first - 0.5 <= cc[0] <= end]
    cs = np.interp([q[0] for q in c], t, s); cv = np.array([q[1] for q in c])
    ev = []
    for key, lab in [('-> sign_arc', 'junction'), ('white line gone', '7A'), ('-> avoid_construction: construction', 'boards'),
                     ('-> drive_to_parking', 'boards done'), ('-> drive_to_lot', 'lot'), ('-> drive_to_level', 'out of lot'),
                     ('-> wait_bar', 'bar'), ('bar lifted', 'bar up'), ('tunnel front', 'tunnel')]:
        hit = [q for q, m in d['/rosout'] if key in m]
        if hit: ev.append((float(np.interp(hit[0], t, s)), lab, hit[0] - first))
    return dict(t=t - first, s=s, v=v, cs=cs, cv=cv, ev=ev, end=end - first)
R = [prep(r) for r in runs]
COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK2, GRID = '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GRID,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2})
fig, ax = plt.subplots(3, 1, figsize=(16, 11), sharex=True, gridspec_kw={'height_ratios': [2.2, 2.2, 1.6]})
labs = [f'{r} ({X["end"]:.1f} s)' for r, X in zip(runs, R)]
ref = R[-1]
for x in ax:
    x.grid(axis='y', color=GRID, lw=0.6)
    for sv, lab, tt in ref['ev']: x.axvline(sv, color='#c9c7bf', lw=0.8, ls=':')
for sv, lab, tt in ref['ev']:
    ax[0].text(sv, 1.01, lab, transform=ax[0].get_xaxis_transform(), rotation=35, fontsize=8, color=INK2, ha='left', va='bottom')
for X, c, l in zip(R, COLS, labs):
    ax[0].plot(X['s'], X['v'], color=c, lw=1.2, label=l)
    ax[1].step(X['cs'], X['cv'], where='post', color=c, lw=1.1, label=l)
ax[0].set_ylabel('measured speed (odom)  m/s'); ax[0].set_ylim(-0.3, 0.32); ax[0].legend(loc='lower left', frameon=False, ncol=len(R))
ax[1].axhline(0.21, color='#d03b3b', lw=0.8, ls='--')
ax[1].set_ylabel('commanded speed  m/s'); ax[1].set_ylim(-0.3, 0.32); ax[1].legend(loc='lower left', frameon=False, ncol=len(R))
A = R[0]; ua = np.maximum.accumulate(A['s'])
x = ax[2]; x.axhline(0, color=COLS[0], lw=1.4); gaps = [0.0]
for X, c, r in zip(R[1:], COLS[1:], runs[1:]):
    grid = np.linspace(0, min(A['s'][-1], X['s'][-1]), 1500)
    gap = np.interp(grid, np.maximum.accumulate(X['s']), X['t']) - np.interp(grid, ua, A['t'])
    x.plot(grid, gap, color=c, lw=1.6); gaps += list(np.percentile(gap, [1, 99]))
    x.text(grid[-1], gap[-1], f'  {r} {gap[-1]:+.1f} s', va='center', fontsize=9, color=INK2)
x.set_ylim(min(-1.5, min(gaps) - 1), max(5, max(gaps) + 1))  # room for a run on other wheels
x.set_ylabel(f'behind {runs[0]}  s'); x.set_xlabel('distance along the run (odom)  m')
fig.suptitle('Speed - ' + ' vs '.join(runs) + ', clocks from the wheels first turning', x=0.02, ha='left', fontsize=13)
plt.subplots_adjust(hspace=0.1, top=0.92)
plt.savefig(out, dpi=80, bbox_inches='tight')
E = [{l: t for s, l, t in X['ev']} for X in R]
order = [l for s, l, t in ref['ev']]
print(f'{"section":24s}' + ''.join(f'{r:>11s}' for r in runs))
prev = None
for l in order:
    if prev and all(prev in e and l in e for e in E):
        print(f'{prev+" -> "+l:24s}' + ''.join(f'{e[l]-e[prev]:11.1f}' for e in E))
    prev = l
print(f'{"total (wheels turning)":24s}' + ''.join(f'{X["end"]:11.1f}' for X in R))

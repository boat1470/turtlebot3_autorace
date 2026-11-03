import sys
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'track.py')).read().split("img=mpimg")[0])
out, path = sys.argv[1], sys.argv[2]
d=load(path); o=d['/odom']
first=[c[0] for c in d['/cmd_vel'] if abs(c[1])>0.01][0]
img=mpimg.imread('/home/boat1470/turtlebot3_ws/src/turtlebot3_autorace/docs/curve_map.png')
fig,ax=plt.subplots(figsize=(13,13)); ax.imshow(img,extent=[0,4000,0,4000])
stages=[(s,m.split('-> ')[1].split(':')[0]) for s,m in d['/rosout'] if m.startswith('stage ') and '->' in m]
cols=plt.cm.tab20.colors
names=[]
for i,(s,n) in enumerate(stages):
    e=stages[i+1][0] if i+1<len(stages) else 1e18
    seg=[world(p[1],p[2]) for p in o if s<=p[0]<=e]
    if len(seg)<2: continue
    if n not in names: names.append(n)
    c=cols[names.index(n)%20]
    xs,ys=zip(*seg); ax.plot(xs,ys,color=c,lw=2.2,label=n if n not in [l.get_label() for l in ax.lines] else None)
keys=[('white line gone','arc'),('arc done','arc end'),('on the spot -','spin'),('park phase 1','back in'),('park phase 2','parked'),('bar down','bar'),('tunnel front','tunnel')]
for s,m in d['/rosout']:
    for k,l in keys:
        if k in m:
            p=min(o,key=lambda q:abs(q[0]-s)); x,y=world(p[1],p[2])
            ax.plot(x,y,'o',color='white',ms=6,mec='k'); ax.annotate(f'{l} {s-first:.1f}s',(x,y),xytext=(6,6),textcoords='offset points',color='white',fontsize=9,bbox=dict(fc='black',alpha=0.6,lw=0))
ax.set_xlim(0,4000); ax.set_ylim(0,4000); ax.legend(loc='upper left',fontsize=9)
ax.set_title(path.split('/')[-1]+'  (odom, scale '+f'{SCALE:.2f}'+')')
plt.savefig(out,dpi=70,bbox_inches='tight')

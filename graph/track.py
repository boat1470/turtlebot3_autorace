import sys, math, bisect, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.image as mpimg
SCALE=1.00
def load(path):
    r=rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=path,storage_id='mcap' if str(path).endswith('.mcap') else ''),rosbag2_py.ConverterOptions('',''))
    types={t.name:t.type for t in r.get_all_topics_and_types()}
    want=[t for t in ('/odom','/detect/lane_state','/detect/lane','/detect/lane_gap','/rosout','/cmd_vel') if t in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=want)); cls={t:get_message(types[t]) for t in want}
    d={t:[] for t in want}
    while r.has_next():
        t,raw,ts=r.read_next(); m=deserialize_message(raw,cls[t]); s=ts*1e-9
        if t=='/odom':
            q=m.pose.pose.orientation; yaw=math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
            d[t].append((s,m.pose.pose.position.x,m.pose.pose.position.y,yaw))
        elif t=='/rosout':
            if m.name=='arx_mission_control': d[t].append((s,m.msg))
        elif t=='/cmd_vel':
            tw=m.twist if hasattr(m,'twist') else m; d[t].append((s,tw.linear.x,tw.angular.z))
        else: d[t].append((s,m.data))
    return d
def world(ox,oy): return 245+1000*SCALE*oy, 1188-1000*SCALE*ox
img=mpimg.imread('/home/boat1470/turtlebot3_ws/src/turtlebot3_autorace/docs/curve_map.png')
fig,ax=plt.subplots(figsize=(12,12)); ax.imshow(img,extent=[0,4000,0,4000])
for p,c in zip(sys.argv[2:],['lime','cyan','magenta','orange','red']):
    d=load(p); o=d['/odom']
    xs,ys=zip(*[world(a[1],a[2]) for a in o]); ax.plot(xs,ys,c,lw=1.2,label=p.split('/')[-1])
ax.set_xlim(1200,2700); ax.set_ylim(900,2100); ax.legend()
plt.savefig(sys.argv[1],dpi=80,bbox_inches='tight')

"""Drive follow_node with fake inputs in an isolated ROS domain and check the deadman."""
import sys, time, math, subprocess, os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
assert os.environ.get('ROS_DOMAIN_ID') == '77', 'run in the isolated domain only'
button = sys.argv[1]
proc = subprocess.Popen(['ros2', 'run', 'mppi', 'follow_node.py', '--ros-args', '-p', 'max_speed:=0.5', '-p', "log_dir:=''",
                         '-p', f'deadman_button:={button}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
rclpy.init(); n = Node('deadman_test')
p_opp = n.create_publisher(Odometry, '/opponent/odom', 10); p_odom = n.create_publisher(Odometry, '/odom', 20)
p_scan = n.create_publisher(LaserScan, '/scan', qos_profile_sensor_data); p_joy = n.create_publisher(Joy, '/joy', 10)
cmds = []
n.create_subscription(AckermannDriveStamped, '/drive', lambda m: cmds.append((time.monotonic(), m.drive.speed)), 10)
def spin(dur, joy=None):
    """publish the fake car inputs for dur seconds; joy = None (no /joy at all) or the index of the held button (-1 = none held)"""
    t_end = time.monotonic() + dur; k = 0
    while time.monotonic() < t_end:
        o = Odometry(); o.header.stamp = n.get_clock().now().to_msg(); o.pose.pose.position.x = 2.0; o.pose.pose.orientation.w = 1.0; p_opp.publish(o)
        e = Odometry(); e.header.stamp = o.header.stamp; p_odom.publish(e)
        if k % 3 == 0:
            s = LaserScan(); s.header.stamp = o.header.stamp; s.angle_min = -2.356; s.angle_increment = 0.0058; s.range_min = 0.05; s.range_max = 25.0
            s.ranges = [5.0] * 811; p_scan.publish(s)
            if joy is not None:
                j = Joy(); j.header.stamp = o.header.stamp; j.axes = [0.0] * 8; j.buttons = [0] * 16
                if joy >= 0: j.buttons[joy] = 1
                p_joy.publish(j)
        k += 1; rclpy.spin_once(n, timeout_sec=0.02)
def last(win=0.4):
    t = time.monotonic(); v = [s for (ts, s) in cmds if t - ts < win]
    return (min(v), max(v)) if v else (None, None)
ok = True
def check(name, cond, got):
    global ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}: /drive speed min..max in the last 0.4 s = {got}")
t_w = time.monotonic()
while not cmds and time.monotonic() - t_w < 12: spin(0.5, None)     # node start-up, no /joy at all
print(f'  node up after {time.monotonic() - t_w:.1f} s'); spin(1.5, None)
if button == '-1':
    g = last(); check('deadman off: drives without any /joy', g[1] is not None and g[1] > 0.45, g)
else:
    g = last(); check('no /joy messages -> stopped', g[0] is not None and g[1] == 0.0, g)
    spin(1.5, -1); g = last(); check('joy alive, nothing held -> stopped', g[1] == 0.0, g)
    spin(1.5, 6); g = last(); check('left bumper (button 6) held, not the trigger -> stopped', g[1] == 0.0, g)
    spin(1.5, 8); g = last(); check('left trigger (button 8) held -> drives at max_speed 0.5', g[0] > 0.45, g)
    t_rel = time.monotonic(); spin(0.5, -1)
    first0 = next((ts - t_rel for ts, s in cmds if ts > t_rel and s == 0.0), None)
    g = last(0.2); check(f'trigger released -> speed 0 after {first0 * 1000:.0f} ms', g[1] == 0.0 and first0 < 0.15, g)
    t_p = time.monotonic(); spin(0.12, 8); ramp = [round(v, 2) for ts, v in cmds if ts > t_p]
    check(f'pressed again -> ramps up from 0 at 2 m/s^2, no jump: {ramp}', ramp and max(ramp) <= 0.30 and max(ramp) > 0.0 and ramp == sorted(ramp), (min(ramp), max(ramp)))
    spin(1.2, 8); t_stale = time.monotonic(); spin(1.0, None)       # trigger still "held" but /joy stops arriving
    first0 = next((ts - t_stale for ts, s in cmds if ts > t_stale and s == 0.0), None)
    g = last(0.3); check(f'/joy stops while held (controller or node gone) -> speed 0 after {first0 * 1000:.0f} ms', g[1] == 0.0 and first0 < 0.45, g)
import signal
os.killpg(proc.pid, signal.SIGINT)
try: out = proc.communicate(timeout=5)[0]
except Exception: os.killpg(proc.pid, signal.SIGKILL); out = proc.communicate()[0]
print('  node log:', ' | '.join(l.split(']: ', 1)[-1][:90] for l in out.splitlines() if ('deadman' in l and 'gap' not in l) or 'Error' in l or 'Traceback' in l)[:500])
if not ok: print(out[-1500:])
print('  RESULT:', 'ALL PASS' if ok else 'FAILED', flush=True); os._exit(0 if ok else 1)

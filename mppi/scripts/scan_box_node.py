#!/usr/bin/env python3
"""Makes the simulated laser see other cars the way the real one does: as their rear box.

The simulator draws every other car into the scan as its whole body (0.58 x 0.31 m). On the real
cars the laser plane is above most of the chassis, and what is guaranteed to be there is the box
the rules require at the back of every car: 12 x 12 cm, from 10 to 30 cm above the ground. An
opponent tracker tuned on whole car bodies would be tuned on something it never gets to see.

    /scan (simulator)  -->  /scan_box  : each opponent's body taken out of the scan (the wall behind
                                         it is put back, by ray casting the map) and its box put in
    /h2h/boxes                          : the boxes as markers, for the viewer (the ego's too)

  opp_odom_topics   true poses of the opponents (the simulator's odometry), to know where the
                    bodies and boxes are. Nothing of this reaches the controller: it only gets the scan
  box_size          [m] side of the box (0.12)
  box_behind_axle   [m] center of the box behind the rear axle (0.065: the box ends with the car)
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.ndimage import distance_transform_edt
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

CAR_L, CAR_W, AXLE_TO_CENTER = 0.58, 0.31, 0.165


def ray_box(ox, oy, dx, dy, cx, cy, yaw, half_l, half_w):
    """distance along each ray (ox, oy) + t * (dx, dy) to a rectangle centered (cx, cy), inf = miss"""
    c, s = np.cos(yaw), np.sin(yaw)
    px, py = (ox - cx) * c + (oy - cy) * s, -(ox - cx) * s + (oy - cy) * c   # origin in the box frame
    rx, ry = dx * c + dy * s, -dx * s + dy * c
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (-half_l - px) / rx, (half_l - px) / rx
        t3, t4 = (-half_w - py) / ry, (half_w - py) / ry
    t_in = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
    t_out = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
    return np.where((t_out >= t_in) & (t_out > 0.0), np.maximum(t_in, 0.0), np.inf)


class ScanBox(Node):
    def __init__(self):
        super().__init__('scan_box_node')
        self.declare_parameter('scan_in', '/scan')
        self.declare_parameter('scan_out', '/scan_box')
        self.declare_parameter('ego_odom_topic', '/ego_racecar/odom')
        self.declare_parameter('opp_odom_topics', ['/opp_racecar/odom'])
        self.declare_parameter('lidar_offset', 0.275)
        self.declare_parameter('box_size', 0.12)
        self.declare_parameter('box_behind_axle', 0.065)
        self.declare_parameter('rate', 40.0)       # [Hz] scans handed on per second at most
        self.declare_parameter('noise_std', 0.01)  # [m] on the box returns, like the simulator's
        gp = lambda k: self.get_parameter(k).value
        self.ego, self.opps = None, {}
        self.dist = None
        self.period, self.last = 1.0 / float(gp('rate')), -1e9
        self.rng = np.random.default_rng(0)
        self.create_subscription(Odometry, gp('ego_odom_topic'), self.ego_cb, 10)
        for i, topic in enumerate(gp('opp_odom_topics')):
            self.create_subscription(Odometry, topic, lambda m, k=i: self.opp_cb(m, k), 10)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(LaserScan, gp('scan_in'), self.scan_cb, 10)
        self.pub = self.create_publisher(LaserScan, gp('scan_out'), 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/h2h/boxes', 1)
        self.get_logger().info(f'{gp("scan_in")} -> {gp("scan_out")}: {len(gp("opp_odom_topics"))} opponent(s) seen as a '
                               f'{gp("box_size")} m box {gp("box_behind_axle")} m behind the rear axle')

    @staticmethod
    def _pose(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        return p.x, p.y, float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)))

    def ego_cb(self, msg):
        self.ego = self._pose(msg)

    def opp_cb(self, msg, k):
        self.opps[k] = self._pose(msg)

    def map_cb(self, msg):
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        self.dist = (distance_transform_edt(~((data >= 50) | (data < 0))) * info.resolution).astype(np.float32)
        self.res, self.ox, self.oy = info.resolution, info.origin.position.x, info.origin.position.y

    def wall_range(self, ox, oy, dx, dy, r_max=25.0):
        """range to the map's walls along each ray (sphere tracing on the wall-distance field)"""
        r = np.full(len(dx), 0.05)
        h, w = self.dist.shape
        for _ in range(80):
            c = np.floor((ox + r * dx - self.ox) / self.res).astype(int)
            row = np.floor((oy + r * dy - self.oy) / self.res).astype(int)
            ok = (row >= 0) & (row < h) & (c >= 0) & (c < w)
            d = np.zeros(len(r))
            d[ok] = self.dist[row[ok], c[ok]]
            done = (d < 0.5 * self.res) | (r >= r_max)
            if done.all():
                break
            r = np.where(done, r, r + np.maximum(0.9 * d, 0.5 * self.res))
        return np.minimum(r, r_max)

    def scan_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last < self.period - 1e-4:
            return
        self.last = now
        gp = lambda k: float(self.get_parameter(k).value)
        if self.ego is None or not self.opps:
            self.pub.publish(msg)
            return
        ex, ey, eyaw = self.ego
        off = gp('lidar_offset')
        ox, oy = ex + off * np.cos(eyaw), ey + off * np.sin(eyaw)
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        ang = eyaw + msg.angle_min + msg.angle_increment * np.arange(len(ranges))
        dx, dy = np.cos(ang), np.sin(ang)
        half, behind = 0.5 * gp('box_size'), gp('box_behind_axle')
        out = ranges.copy()
        boxes = []
        for x, y, yaw in self.opps.values():
            # the body the simulator drew: take it out, put the wall behind it back
            # (every return that ENDS on the body, a little inflated: comparing ranges with the
            # ray's entry point missed the returns that graze a corner, and those leftovers made
            # a second small cluster next to the box)
            cx, cy = x + AXLE_TO_CENTER * np.cos(yaw), y + AXLE_TO_CENTER * np.sin(yaw)
            fin = np.isfinite(ranges)
            r0 = np.where(fin, ranges, 0.0)
            hx, hy = ox + r0 * dx - cx, oy + r0 * dy - cy
            lx, ly = hx * np.cos(yaw) + hy * np.sin(yaw), -hx * np.sin(yaw) + hy * np.cos(yaw)
            on_body = fin & (np.abs(lx) < 0.5 * CAR_L + 0.08) & (np.abs(ly) < 0.5 * CAR_W + 0.08)
            if on_body.any():
                if self.dist is not None:
                    out[on_body] = self.wall_range(ox, oy, dx[on_body], dy[on_body])
                else:
                    k = np.arange(len(out))
                    good = ~on_body & np.isfinite(out)
                    out[on_body] = np.interp(k[on_body], k[good], out[good])
            bx, by = x - behind * np.cos(yaw), y - behind * np.sin(yaw)
            boxes.append((bx, by, yaw))
        for bx, by, yaw in boxes:  # then the boxes, in front of whatever is behind them
            t_box = ray_box(ox, oy, dx, dy, bx, by, yaw, half, half)
            hit = np.isfinite(t_box) & (t_box < out)
            out[hit] = t_box[hit] + self.rng.normal(0.0, gp('noise_std'), int(hit.sum()))
        msg.ranges = out.astype(np.float32).tolist()
        self.pub.publish(msg)

        arr = MarkerArray()
        for i, (bx, by, yaw) in enumerate(boxes + [(ex - behind * np.cos(eyaw), ey - behind * np.sin(eyaw), eyaw)]):
            m = Marker()
            m.header.frame_id, m.header.stamp = 'map', msg.header.stamp
            m.ns, m.id, m.type, m.action = 'box', i, Marker.CUBE, Marker.ADD
            m.pose.position = Point(x=float(bx), y=float(by), z=0.2)
            m.pose.orientation.z, m.pose.orientation.w = float(np.sin(0.5 * yaw)), float(np.cos(0.5 * yaw))
            m.scale.x = m.scale.y = 2.0 * half
            m.scale.z = 0.2
            m.color = ColorRGBA(r=0.1, g=0.1, b=0.1, a=0.9)
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ScanBox()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

"""Head-to-head in the simulator on the race 2 map: simulator + ego MPPI + opponent cars.

    ros2 launch mppi h2h_sim_launch.py                       # ego vs one slow opponent
    ros2 launch mppi h2h_sim_launch.py num_agents:=1         # ego alone (lap-time check)
    ros2 launch mppi h2h_sim_launch.py num_agents:=4         # 4-car race, as in Race 3
    ros2 launch mppi h2h_sim_launch.py opp_speed_scale:=0.9 opp_offset:=-0.3
    ros2 launch mppi h2h_sim_launch.py open_foxglove:=true   # with the viewer
    ros2 launch mppi h2h_sim_launch.py h2h:=false            # the race 2 controller (baseline)
    ros2 launch mppi h2h_sim_launch.py latency:=false        # ideal simulator, no delays
    ros2 launch mppi h2h_sim_launch.py fused_pose:=false     # controller on the late particle-filter pose
    ros2 launch mppi h2h_sim_launch.py raceline:=race2_raceline_opt.csv pace:=race

Switchback test: two equal cars (same controller, same pace), the opponent on the early-apex
line (covers the inside of every corner), the ego 1.5 m behind it with the late-apex line:
    ros2 launch mppi h2h_sim_launch.py latency:=false pace:=race raceline:=race2_raceline_opt.csv \\
        opp_controller:=mppi opp_line:=race2_raceline_opt_earlyapex.csv start_gap:=1.5 \\
        switchback:=race2_raceline_opt_apex.json

Run it from a shell with the workspace venv activated (mppi_node.py needs jax).

The ego runs the REAL-CAR parameters of car_launch.py (read from that file, so the two cannot
drift apart) with only what the simulator forces changed: topics, the map-frame raceline and the
lidar position. latency_node.py puts the car's delays (old pose, slow control rate, actuation
lag) between the simulator and the controller, so the car's latency compensation stays as it is;
with latency:=false the delays and the compensation are both off.
"""
import importlib.util
import json
import os
import tempfile
import time

import numpy as np
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RACELINE = 'race2_raceline.csv'  # in the race2_map_edited frame

SIM_OVERRIDES = {
    'use_sim': True,
    'odom_topic': '/ego_racecar/odom',
    'drive_topic': '/drive',
    'lidar_offset': 0.275,     # lidar_base_link_to_lidar_tf in sim_race2.yaml
    'recovery': True,          # back up when stopped at a wall (scripts/stuck_test.py)
    'yaw_gain': 0.87,          # measured: the simulated car turns 0.87 of the bicycle model's yaw rate in corners
}
# pace:=race: the speed profile of the team's final race 2 raceline (scripts/raceline.py) instead of
# the cautious one of car_launch.py (a_lat 1.69: so slow that the shortest line is the fastest)
RACE_PACE = {'a_lat': 4.3, 'a_lon': 2.5, 'v_min': 1.0, 'v_max': 4.0, 'max_speed': 5.0,
             'slow_ignore_walls': True,   # walls are not obstacles to brake for
             # The obstacle grid from the map + what the scan sees that is not on it, instead of
             # "what the scan has not seen is blocked". With the latter the track behind every
             # corner is an obstacle until the car sees it: the detour planner moved the reference
             # up to 1 m off the raceline and the narrow-gap logic braked for it, on an empty
             # track (8.65 -> 7.72 s per lap with the car's delays, 7.70 -> 7.41 s without).
             'use_map_prior': True}
# latency:=true (default): the controller talks to the simulator through latency_node.py
DELAYED_TOPICS = {
    'speed_cmd_predicted': True,   # the speed command from the speed the car will have on arrival
    'odom_topic': '/sim_delayed/odom',
    'scan_topic': '/sim_delayed/scan',
    'drive_topic': '/mppi/drive',
}
# fused_pose:=true (default, needs latency:=true): pose_fusion_node.py brings the late pose up to
# now with speed + gyro, so only the control step and the steering response are left to compensate
FUSED_POSE = {
    'odom_topic': '/fused/odom',
    'latency_comp': 0.19,      # car: 0.045 control step + 0.125 steering + ~0.02 sensor age
}
FUSION_NODE = {
    'pf_topic': '/sim_delayed/odom',
    'odom_topic': '/sim/vesc_odom',
    'imu_topic': '/sim/imu',
    'lag': 0.12,               # pose_delay of latency_node.py
    'lever': 0.0,              # the simulator's pose point is the rear axle
    'rate': 22.0,              # the car's controller manages ~22 steps per second (45 ms each)
}
# latency:=false: nothing but the compute time to compensate
NO_LATENCY = {
    'latency_comp': -1.0,
    'latency_history': False,
}


def _car_parameters():
    """PARAMETERS of car_launch.py, the configuration that ran on the car"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'car_launch.py')
    spec = importlib.util.spec_from_file_location('car_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.PARAMETERS)


def _start_config(share, raceline, start_s, gap, n):
    """sim config with the cars lined up on the raceline: the ego at station start_s, every
    opponent `gap` m further along. Returns the path of a temporary yaml."""
    with open(os.path.join(share, 'config', 'sim_race2.yaml')) as f:
        cfg = yaml.safe_load(f)
    xy = np.loadtxt(os.path.join(share, 'waypoints', raceline), delimiter=',', comments='#', ndmin=2)[:, :2]
    xy_c = np.vstack([xy, xy[:1]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy_c, axis=0).T))]
    p = cfg['bridge']['ros__parameters']
    for i in range(n):
        si = (start_s + i * gap) % s[-1]
        x, y = np.interp(si, s, xy_c[:, 0]), np.interp(si, s, xy_c[:, 1])
        x2, y2 = np.interp((si + 0.2) % s[-1], s, xy_c[:, 0]), np.interp((si + 0.2) % s[-1], s, xy_c[:, 1])
        k = '' if i == 0 else str(i)
        p['sx' + k], p['sy' + k], p['stheta' + k] = float(x), float(y), float(np.arctan2(y2 - y, x2 - x))
    path = os.path.join(tempfile.gettempdir(), f'h2h_sim_{os.getpid()}.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return path


def _setup(context):
    share = get_package_share_directory('mppi')
    n = int(LaunchConfiguration('num_agents').perform(context))
    speed = float(LaunchConfiguration('opp_speed_scale').perform(context))
    offset = float(LaunchConfiguration('opp_offset').perform(context))
    run_ego = LaunchConfiguration('ego').perform(context).lower() == 'true'
    latency = LaunchConfiguration('latency').perform(context).lower() == 'true'
    h2h = LaunchConfiguration('h2h').perform(context).lower() == 'true'

    lc = lambda k: LaunchConfiguration(k).perform(context)
    raceline, pace = lc('raceline'), lc('pace')
    config = os.path.join(share, 'config', 'sim_race2.yaml')
    # sync_start: every car stands until the same instant, this many seconds from now
    start_time = time.time() + float(lc('sync_start')) if lc('sync_start') else 0.0
    if lc('start_gap'):
        config = _start_config(share, raceline, float(lc('start_s')), float(lc('start_gap')), n)

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py')),
        launch_arguments={
            'config': config,
            'map_path': os.path.join(share, 'maps', lc('map')),
            'num_agents': str(n),
            'open_foxglove': LaunchConfiguration('open_foxglove'),
        }.items())

    nodes = [sim]
    if run_ego:
        ego = _car_parameters()
        ego.update(SIM_OVERRIDES)
        ego['waypoint_file'] = LaunchConfiguration('raceline').perform(context)
        if LaunchConfiguration('pace').perform(context) == 'race':
            ego.update(RACE_PACE)
        ego.update(DELAYED_TOPICS if latency else NO_LATENCY)
        if not latency:
            # nothing between a command and its effect: the ego can sit closer behind a car
            ego.update({'h2h_follow_decel': 2.0, 'h2h_react_time': 0.1, 'h2h_follow_time': 0.1})
        ego['h2h'] = h2h
        # the opponent's controller comes up later than the ego's: wait for it (or, with
        # sync_start, both wait for the same instant)
        ego['h2h_start_hold'] = 0.0 if start_time else 4.0
        ego['start_time'] = start_time
        ego['h2h_pass'] = LaunchConfiguration('h2h_pass').perform(context).lower() == 'true'
        ego['h2h_log'] = LaunchConfiguration('h2h_log').perform(context)
        ego['h2h_lines'] = lc('switchback')
        ego['h2h_sb_corners'] = lc('sb_corners')
        ego['h2h_sb_commit'] = lc('sb_commit') == 'true'
        # box:=true (default): the ego's laser sees the opponents as their rear boxes (scan_box_node.py)
        box = lc('box') == 'true' and n > 1
        if box:
            nodes.append(Node(package='mppi', executable='scan_box_node.py', output='screen', parameters=[{
                'opp_odom_topics': [f'/opp_racecar{"" if i == 1 else i}/odom' for i in range(1, n)]}]))
            if not latency:   # with the delays, latency_node.py takes /scan_box and hands it on
                ego['scan_topic'] = '/scan_box'
        ego['h2h_box'] = box
        if latency:
            nodes.append(Node(package='mppi', executable='latency_node.py', output='screen',
                              parameters=[dict({'scan_in': '/scan_box' if box else '/scan'},
                                               **json.loads(lc('latency_params') or '{}'))]))
            if LaunchConfiguration('fused_pose').perform(context).lower() == 'true':
                ego.update(FUSED_POSE)
                nodes.append(Node(package='mppi', executable='pose_fusion_node.py', output='screen',
                                  parameters=[dict(FUSION_NODE, **json.loads(lc('fusion_params') or '{}'))]))
            else:
                ego['h2h_pose_lag'] = FUSION_NODE['lag']  # the tracker must know the pose is old
        ego['visualize'] = LaunchConfiguration('visualize').perform(context).lower() == 'true'
        # last, so they also override what the presets above set (they used to come before
        # FUSED_POSE: a latency_comp given on the command line was silently replaced by 0.19)
        ego.update(json.loads(lc('car_params') or '{}'))   # both cars
        ego.update(json.loads(lc('ego_params') or '{}'))   # the ego only
        nodes.append(Node(package='mppi', executable='mppi_node.py', output='screen',
                          parameters=[ego]))
    for i in range(1, n):
        suffix = '' if i == 1 else str(i)  # the bridge's naming: opp_racecar, opp_racecar2, ...
        if lc('opp_controller') == 'mppi':
            # an equal car: the same MPPI, speed profile and obstacle avoidance as the ego, on its
            # own line, without the head-to-head logic. (Without the obstacle avoidance it is not
            # equal: the ego's slows a little near the walls, and the opponent pulled 15 m away in
            # 50 s on the same raceline.) Always without the delays.
            opp = _car_parameters()
            opp.update(SIM_OVERRIDES)
            opp.update(NO_LATENCY)
            if pace == 'race':
                opp.update(RACE_PACE)
            opp.update(json.loads(lc('car_params') or '{}'))
            if lc('opp_slow_zone'):
                opp['slow_zone'] = [float(q) for q in lc('opp_slow_zone').split(',')]
            opp.update({'line_set': lc('opp_line_set'), 'line_key': 'defend', 'line_corners': lc('opp_line_corners'),
                        'start_time': start_time + float(lc('opp_start_delay')) if start_time else 0.0,
                        'start_offset': float(lc('opp_start_delay'))})
            opp.update({'waypoint_file': lc('opp_line') or raceline, 'odom_topic': f'/opp_racecar{suffix}/odom',
                        'drive_topic': f'/opp_drive{suffix}', 'scan_topic': f'/opp_scan{suffix}',
                        'h2h': False, 'visualize': False,
                        'speed_scale': speed if lc('opp_speed_scale_set') == 'true' else 1.0})
            nodes.append(Node(package='mppi', executable='mppi_node.py', name=f'opp_mppi{suffix}',
                              output='log', parameters=[opp]))
            continue
        nodes.append(Node(
            package='mppi', executable='opponent_node.py', name=f'opponent{suffix}', output='screen',
            parameters=[{
                'odom_topic': f'/opp_racecar{suffix}/odom',
                'drive_topic': f'/opp_drive{suffix}',
                'waypoint_file': lc('opp_line') or RACELINE,
                'speed_scale': speed,
                'lateral_offset': offset,
            }]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('num_agents', default_value='2', description='cars on track, ego included'),
        DeclareLaunchArgument('opp_speed_scale', default_value='0.6',
                              description='opponent speed = raceline csv speed x this'),
        DeclareLaunchArgument('opp_offset', default_value='0.0',
                              description='opponent sideways offset from the raceline [m], + = left'),
        DeclareLaunchArgument('ego', default_value='true', description='false: do not start the ego MPPI'),
        DeclareLaunchArgument('visualize', default_value='false', description='MPPI rollout markers'),
        DeclareLaunchArgument('raceline', default_value=RACELINE, description="the ego's raceline csv"),
        DeclareLaunchArgument('pace', default_value='car',
                              description="'car' = speed profile of car_launch.py, 'race' = a_lat 4.3, v_max 4"),
        DeclareLaunchArgument('opp_controller', default_value='pursuit',
                              description="'pursuit' = simple follower of a line, 'mppi' = an equal car"),
        DeclareLaunchArgument('opp_line', default_value='', description="the opponents' line csv ('' = default)"),
        DeclareLaunchArgument('opp_speed_scale_set', default_value='false',
                              description='true: opp_speed_scale also applies to an mppi opponent'),
        DeclareLaunchArgument('start_gap', default_value='',
                              description="[m] line the cars up on the ego's raceline this far apart ('' = poses of the sim config)"),
        DeclareLaunchArgument('start_s', default_value='20.0', description='[m] station of the ego at the start'),
        DeclareLaunchArgument('map', default_value='race2_map_edited', description='map in mppi/maps (without .yaml)'),
        DeclareLaunchArgument('sync_start', default_value='',
                              description="[s] all cars set off together this long after the launch ('' = each when it is up)"),
        DeclareLaunchArgument('opp_slow_zone', default_value='',
                              description="'from,to,factor': mppi opponents drive stations from .. to at factor x their speed"),
        DeclareLaunchArgument('opp_start_delay', default_value='0.0',
                              description='[s] with sync_start: mppi opponents set off this much later than the ego'),
        DeclareLaunchArgument('opp_line_set', default_value='',
                              description="mppi opponents drive the defender's (early-apex) line of this json of make_apex_lines.py"),
        DeclareLaunchArgument('opp_line_corners', default_value='',
                              description="... only in these corners, e.g. 'tight_mid' ('' = every corner)"),
        DeclareLaunchArgument('switchback', default_value='',
                              description="late-apex lines json of scripts/make_apex_lines.py ('' = no switchback)"),
        DeclareLaunchArgument('box', default_value='true',
                              description='the laser sees opponents as their 12 cm rear box, not as whole car bodies'),
        DeclareLaunchArgument('car_params', default_value='',
                              description='json of mppi_node parameters for the ego AND mppi opponents, e.g. {"a_lat": 5.5}'),
        DeclareLaunchArgument('ego_params', default_value='', description='the same, for the ego only'),
        DeclareLaunchArgument('sb_corners', default_value='', description="switchback only in these corners, e.g. 'corner2'"),
        DeclareLaunchArgument('sb_commit', default_value='false', description='true: a started switchback is never given up'),
        DeclareLaunchArgument('latency', default_value='true',
                              description="inject the real car's delays (latency_node.py)"),
        DeclareLaunchArgument('fused_pose', default_value='true',
                              description='high-rate pose from speed + gyro (pose_fusion_node.py)'),
        DeclareLaunchArgument('h2h', default_value='true',
                              description='opponent tracking + follow / pass; false = race 2 controller'),
        DeclareLaunchArgument('h2h_pass', default_value='true', description='false: follow only'),
        DeclareLaunchArgument('h2h_log', default_value='', description='csv path: one line per control step'),
        DeclareLaunchArgument('open_foxglove', default_value='false'),
        DeclareLaunchArgument('latency_params', default_value='',
                              description='json: parameters of latency_node.py, e.g. {"actuation_delay": 0.0}'),
        DeclareLaunchArgument('fusion_params', default_value='',
                              description='json: parameters of pose_fusion_node.py, e.g. {"rate": 50.0}'),
        OpaqueFunction(function=_setup),
    ])

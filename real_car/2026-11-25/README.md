# Real car, 2026-11-25: camera + LiDAR opponent EKF and the follow experiment

First day of the opponent EKF (`mppi/scripts/opponent_ekf_node.py`, lab 10 car observations + VESC
speed + gyro) on the car. This folder holds the recordings, the calibration, the resource
measurements and the scripts used, so the work can continue in the simulator without the car.

Everything here was measured on car 5 (Jetson Orin Nano Super, SICK TiM LiDAR at 15 Hz, RealSense
D435i colour camera, VESC) in the lab: floor and table top, straight lines only, no localization,
no MPPI, no particle filter running.

## Results in short

| | |
|---|---|
| Detection, opponent 0.25-2.5 m, driving up to 1.36 m/s | 1304 of 1311 frames (99 %), confidence 0.94 |
| EKF position vs. the detections | 2.4 cm rms |
| EKF speed along the ego axis, opponent moving | +0.68 m/s where the detections give +0.70; -0.67 where they give -0.70 |
| Observation age when it reaches the EKF | 60 ms mean, 72 ms p95 (camera + 37 ms inference + DDS) |
| Following (two runs, up to 1.0 and 1.8 m/s) | gap 0.80 +- 0.18 m and 0.74 +- 0.13 m while moving, reference 0.5 m |
| CPU, perception + EKF on top of the car stack | 2.1 of 6 cores as launched; 1.3 cores with `OPENBLAS_NUM_THREADS=1` |
| GPU, perception | 29 % mean, bursts to 98 % |

Three things had to be fixed or found first:

1. **The camera is not aligned with the LiDAR** (2.76 deg pitch, 1.83 deg yaw). With the ideal
   rotation matrix the scan was drawn 33 px too high in the image, the LiDAR point fell above the
   detection box, and the observation was dropped beyond about 1 m. See [Calibration](#calibration).
2. **`/drive` has no deadman on this car.** `ackermann_mux` passes it without any button (the
   autonomous deadman of `joy_teleop.yaml` publishes to `/dev/null`). `follow_node` now has its own:
   left trigger, joy button 8.
3. **The EKF node burns 1.25 cores**, more than the detector, because OpenBLAS spins up threads for
   7x7 matrices. One environment variable brings it to 0.37. See [Resources](#resources-and-latency).

## Recordings

`bags/` are the original rosbag2 (mcap) files, `npz/` the same runs as numpy files that need no ROS
(format: docstring of [tools/export_bag_npz.py](tools/export_bag_npz.py)). The bags have no camera
images: the detector ran on the car and only its output (`/perception/car_observations`) was recorded.

| Run | Length | What happened | Detection |
|---|---|---|---|
| `opp_static_1125_1337` | 34 s | both cars standing, opponent 0.68 m ahead. **Before** the calibration | 91 % |
| `opp_static_1125_1346` | 112 s | opponent moved between 0.65 and 2.3 m. **Before** the calibration | 92 % at 0.6-0.8 m, 3-22 % beyond 1 m |
| `opp_1125_142116` | 48 s | calibrated. Both standing, opponent 2.65 m ahead, 6 deg left | 100 % |
| `opp_1125_142916` | 97 s | calibrated. Ego standing, opponent driven back and forth, 0.25-2.5 m, up to 1.36 m/s | 99 % |
| `opp_1125_143405` | 52 s | calibrated. Follow run with `/drive`, ego up to 1.0 m/s (log `follow_20261125_143412.csv`) | 100 % |

Topics: `/perception/car_observations`, `/opponent/odom`, `/opponent/observations_ego`, `/scan`, `/odom`,
`/sensors/imu/raw`, `/sensors/core`, `/tf`, `/tf_static`, `/rosout`, and `/drive` in the last run.
`/opponent/odom` is what the EKF published on the car with the launch parameters of that day
(`motion:=straight`, `pose_topic:=none`).

The opponent was a second F1TENTH chassis without body. During the afternoon an open cardboard box
(12 cm wide, 21 cm tall) was put on it; the data does not say for which runs.

`follow_logs/` has the CSV of three follow runs (columns: docstring of `mppi/scripts/follow_node.py`):

| Log | States | Ego |
|---|---|---|
| `follow_20261125_132643.csv` (128 s) | wait 87 %, estop_lidar 13 % | never moved: before the calibration, no track |
| `follow_20261125_143412.csv` (43 s) | follow 56 %, deadman 31 %, estop_lidar 12 % | up to 1.01 m/s, gap 0.80 +- 0.18 m while moving |
| `follow_20261125_144128.csv` (33 s) | follow 66 %, deadman 16 %, estop_lidar 18 % | up to 1.83 m/s, gap 0.74 +- 0.13 m while moving |

There is no bag of the 14:41 run. It ended with a LiDAR emergency stop from 0.95 m/s; during the
stop the LiDAR gap was 0.00-0.15 m (median 0.02) and the filter gap went down to -0.08 m, so the ego
stopped within a few centimetres of the opponent, possibly touching it.

## Calibration

`vision_lab` (lab 10) projects the scan into the image with the camera position measured by tape,
c_L = (0.071, 0.038, -0.0345) m, and an ideal axis swap as rotation. A detection is published only
if a projected LiDAR point lies inside its box.

A cardboard box was placed alone on the floor at 1.14 m and at 2.69 m (`calib/d1.png`, `d2.png`
with the matching scans `d1.npz`, `d2.npz`):

| | 1.14 m | 2.69 m |
|---|---|---|
| Box in the image, columns | 485-562 | 481-513 |
| Box in the image, rows | 241.5-379.5 | 281.5-337 |
| Scan on the box, ideal matrix (columns / row) | 461-549 / 252 | 459-489 / 266 |
| Inside the box? | yes, 10 px below its top | **no, 16 px above its top** |
| Scan on the box, calibrated (columns / row) | 483-571 / 285.5 | 481-511 / 299 |

The error is the same number of pixels at both distances (33 px vertical, 21-23 px horizontal), so
it is a rotation, not a wrong lens position. Fit (`tools/calib_fit.py`, residual 0.2 px): pitch
2.76 deg, yaw 1.83 deg; camera 10.9 cm and LiDAR plane 14.4 cm above the floor.

With the ideal matrix the opponent has to reach this far above the LiDAR plane for a match: 3 cm at
0.65 m, 5 cm at 1.1 m, 10 cm at 2 m, 14 cm at 3 m. That is why it was only detected when close.

```python
ROTATION_LIDAR_TO_CAMERA_OPTICAL = [
    0.031940, -0.999490,  0.000000,
    0.048102,  0.001537, -0.998841,
    0.998332,  0.031903,  0.048127,
]
```

`calib/before_after.jpg` shows the scan on the box before (red) and after (green).
**The matrix lives in the lab 10 repository** (`vision_lab/launch/car_observations_launch.py`, branch
`car/jetson`) and is only changed on the car, not pushed there.
`calib/lab10_launch_calibration.patch` is that change (`git apply` in the lab 10 repository).

Assumption: the LiDAR scan plane is parallel to the floor. Check: the plane should be 14.4 cm above
the floor. Verified only with the box and then with the opponent at 2.65 m (6 % -> 100 %).

## EKF behaviour

`python3 tools/replay_ekf_npz.py npz/<run>.npz` replays a run through `opponent_ekf.py` and compares
it with the detections themselves (position: surface point + offsets; speed: difference over 6
frames plus the ego speed).

Opponent moving (`opp_1125_142916`, ego standing): position 2.4 cm rms; speed along the ego axis
+0.68 m/s where the reference says +0.70 (12 s), -0.67 where it says -0.70 (12 s), 0.00 standing
(50 s); about 0.2-0.3 s late when the opponent starts. Sample by sample the speed differs from the
(itself noisy) reference by 0.27 m/s rms.

When the opponent drives toward the ego the filter flips its heading by 180 deg (it keeps the speed
positive). The speed along the ego axis keeps the right sign; the heading in the log jumps between
0 and +-180 deg.

Both cars standing (`opp_static_1125_1337`): the heading wanders (up to 104 deg in 30 s) because the
opponent's yaw rate is a random walk that nothing observes while it stands, and the detections jump
6 cm sideways between two boxes on the same car. Together this shows up as speed: up to 0.33 m/s
along the ego axis. `follow_node` would command up to 0.57 m/s toward a standing car at the
reference gap.

Suggested for `motion:=straight`, **not applied** on the car (`SUGGESTED` in `replay_ekf_npz.py`):

| Parameter | Launch | Suggested |
|---|---|---|
| `ekf.opp_yaw_accel_std` | 0.05 | 0.0 |
| `ekf.init_yaw_rate_std` | 0.1 | 0.001 |
| `ekf.opp_heading_std` | 0.01 | 0.02 |
| `ekf.range_std` | 0.05 | 0.02 |
| `min_track_speed` (follow_node) | 0.0 | 0.1 |

`python3 tools/sweep_follow_sim.py npz/opp_static_1125_1337.npz` (33 s): the standing recording
replayed, and a closed-loop straight-line follow simulation fed with the recorded detection noise.
The simulated motor is an assumed 0.15 s lag, so compare the rows, not the absolute numbers.

| Parameters | Standing: follow command max | Standing: heading max | Sim: speed error rms | Sim: minimum gap | Sim: creep while standing |
|---|---|---|---|---|---|
| launch (`motion:=straight`) | 0.57 m/s | 104 deg | 0.201 m/s | 0.070 m | 0.027 m/s |
| suggested EKF parameters | 0.15 m/s | 3 deg | 0.174 m/s | 0.090 m | 0.045 m/s |
| suggested + `min_track_speed` 0.1 | 0.02 m/s | 3 deg | 0.174 m/s | 0.090 m | 0.006 m/s |

No track was lost in any case, also with the opponent weaving +-0.30 m. On the moving recordings the
suggested parameters change nothing measurable (position 2.3 cm, speed 0.27 m/s rms in both).

## Follow runs

`follow_node` holds a gap with a PD law on the EKF output; steering is pure pursuit on the opponent.
What the logs show besides the table above:

- **The LiDAR gap is smaller than the filter gap**: 5 cm (median) in the first run, 13 cm in the
  second, and up to 26 cm when close. The filter gap assumes the surface the camera picked is the
  rear end of the opponent (`surface_offset` 0.28, `opp_rear_to_center` 0.28). The nearest return in
  the corridor says the opponent ends closer than that: the LiDAR plane, 14.4 cm up, also hits the
  rear of the chassis, not only the box above the car centre. `gap_ref` 0.5 m was 0.25-0.45 m in
  reality.
- **Stray LiDAR returns** at 0.03-0.07 m inside the corridor ahead (steep angles, single beams) give
  a LiDAR gap below zero (seen: -0.17 m) and can trigger `estop_lidar` with nothing there.
  `scan_cb` has no lower range limit beyond `range_min`.
- **Braking margin**: see the end of the 14:41 run above. `tools/sweep_follow_sim.py` gives 6-9 cm
  minimum gap for an opponent braking at 6 m/s^2 from 1.2 m/s, for every parameter set.

## Deadman

`follow_node` publishes a non-zero command only while joy button `deadman_button` is held (8 = left
trigger on the 8BitDo Ultimate 2C; 6 = left bumper is the teleop deadman and still overrides).
Released, or no `/joy` for 0.3 s: speed 0 at once; pressed again: ramp from 0.
`tools/test_deadman.py` checks this with fake inputs in an isolated ROS domain (8 checks pass). If
the controller goes out of Bluetooth range without disconnecting, `/joy` may keep repeating the last
state; the teleop deadman has the same limit.

## Resources and latency

Jetson Orin Nano Super, 25 W mode, 6 cores (schedutil, 730-1344 MHz), 7.5 GB RAM shared with the
GPU. Car standing, 45 s per case. CPU is in percent of one core: 600 = the whole machine. Raw
samples and tegrastats logs are in `resources/`.

| Case | CPU all | Busiest core | GPU | RAM used | Board power |
|---|---|---|---|---|---|
| Car stack only (LiDAR, VESC, joy, mux) | 36 % | 13 % | 0 % | 3.58 GB | 5.0 W |
| + perception + EKF, as launched | 257 % | 82 % (max 98) | 29 % (p95 98) | 4.04 GB | 8.8 W |
| + perception + EKF, `OPENBLAS_NUM_THREADS=1` | 180 % | 51 % (max 73) | 31 % (p95 98) | 4.02 GB | 8.6 W |

Both loaded cases include about 15 % for the latency probe that was subscribing at the same time.

| Process | CPU | Memory (RSS) |
|---|---|---|
| `car_observations.py` (camera + RT-DETRv2 FP16 TensorRT) | 67 % (73 % in the second case) | 969 MB |
| `opponent_ekf_node.py`, as launched | **125 %** | 82 MB |
| `opponent_ekf_node.py`, `OPENBLAS_NUM_THREADS=1` | **37 %** | 80 MB |
| `follow_node.py` (50 Hz, fake inputs, isolated domain) | 22 % | 61 MB |
| LiDAR driver / VESC nodes + mux / joy | 2 % / 10 % / 5 % | 41 / 153 / 113 MB |

The EKF is plain numpy on 7x7 matrices, at 50 Hz speed + 50 Hz gyro + 15 Hz detections that arrive
late and make it replay its history. numpy's OpenBLAS starts one thread per core for these and they
spin; one thread does the same work with a third of the CPU and the same timing
(`/opponent/odom` 50.2 Hz, period p95 21 ms in both cases). Not applied: set it for the node
(`additional_env={'OPENBLAS_NUM_THREADS': '1'}` in the launch file) or in the shell.

Perception, per frame (`resources/perception_stages.json`, 300 frames):

| Stage | Mean | p95 |
|---|---|---|
| `camera.read()` (V4L2 dequeue) | 1.3 ms | 1.7 ms |
| `detector.detect()` (preprocess + inference + postprocess) | 36.6 ms | 38.5 ms |
| `project_scan()` | 0.6 ms | 0.7 ms |
| `estimate_position()` | 0.4 ms | 0.4 ms |

Free running the node could do 26 Hz; it publishes one observation per scan, 15 Hz.

Delays at a subscriber (`resources/perception_ekf_latency.json`):

| | Mean | p95 | Max |
|---|---|---|---|
| Observation age = receive time - scan stamp | 60 ms | 72 ms | 126 ms |
| Camera frame - scan stamp | 11 ms | | |
| `/opponent/odom` period | 19.9 ms | 21.1 ms | 27.6 ms |

The EKF inserts each observation at its beam time and predicts to now, so its output is not 60 ms
old, but it has not seen the last 60 ms of what the opponent did.

Camera: `/dev/video4`, the D435i colour stream through V4L2 at 960x540 YUYV, 60 fps (about 62 MB/s
over USB 3). Only one process can open it. The depth stream and the RealSense SDK are not used.

**Not measured**: the same with MPPI and the particle filter running. MPPI drives the car, so it was
not started in the lab. Both use the GPU (JAX, `range_libc`), and the detector already fills it in
bursts, so the 37 ms inference and with it the 60 ms will grow. To measure on the track:

```bash
python3 tools/measure_resources.py --duration 60 --out resources/race_stack \
    --match perception=lib/vision_lab/car_observations.py --match ekf=lib/mppi/opponent_ekf_node.py \
    --match mppi=lib/mppi/mppi_node.py --match pf=particle_filter
python3 tools/measure_latency.py --duration 60 --out resources/race_stack_latency.json
```

## For the switchback study

The switchback uses the LiDAR-only tracker (`h2h.OpponentTracker`: map frame, needs the map and a
pose). What these recordings can and cannot tell about replacing or backing it with the camera EKF:

- They cover **straight lines only**: bearings between -16 and +20 deg, ranges 0.25-2.7 m, opponent
  up to 1.4 m/s, ego up to 1.8 m/s. No corner, no localization, no map-frame output
  (`/opponent/odom_world` needs `pose_topic`).
- The camera sees +-34.8 deg (960 px at fx 691.5); `max_bearing` is 0.7 rad, so the camera is the
  limit. A detection also needs a LiDAR return in the centre column of its box.
- Without accepted detections the filter predicts (constant turn rate and speed) and drops the track
  after `ekf.lost_timeout` = 1.0 s; a new track needs 3 hits in 0.5 s.
- Each bag also has the full `/scan`, so the LiDAR tracker can be run on the same seconds and
  compared with the camera EKF.

To take the opponent out of view in a replay:

```bash
python3 tools/replay_ekf_npz.py npz/opp_1125_142916.npz --max-bearing 0.15   # narrower camera
```

or from Python, `replay(d, params, drop=lambda t, x, y: 30.0 < t < 31.5)` in `replay_ekf_npz.py`
removes the detections of a time window (an occlusion); `events()` gives the measurements in arrival
order for any other filter.

For the simulator, the numbers to reproduce: 15 Hz, one surface point per detected car, 60 ms old on
arrival (72 ms p95), 1-9 % of frames without a car, a second point 6 cm beside the first in up to a
third of the frames (`det_*` in the npz files have the real sequences), +-34.8 deg field of view.

## Files

| | |
|---|---|
| `bags/` | rosbag2 (mcap), 67 MB |
| `npz/` | the same runs for numpy, 8 MB |
| `follow_logs/` | `follow_node` CSV logs |
| `calib/` | box images and scans at two distances, before/after picture, lab 10 patch |
| `resources/` | CPU / GPU / memory samples, tegrastats logs, latency, stage timing, `system_info.txt` |
| `car/setup_ros2.bash` | the launcher used on the car (`opp-ekf`, `follow`, `opp-bag`, ...); controller address removed |
| `tools/` | below |

Tools that need only numpy (run from `tools/`):

| | |
|---|---|
| `replay_ekf_npz.py` | replay a run through the EKF, compare with the detections and with the car's output |
| `detection_stats.py` | detection rate by distance, confidence, bearing, observation age |
| `sweep_follow_sim.py` | parameter sets on the standing recording and in a closed-loop follow simulation |
| `calib_fit.py` | the calibration fit |
| `measure_resources.py` | CPU / GPU / memory sampler (Jetson: tegrastats; also needs psutil) |

Tools for the car (ROS 2 Jazzy and the workspace sourced):

| | |
|---|---|
| `export_bag_npz.py` | bag -> npz |
| `measure_latency.py` | rates and delays of the perception -> EKF chain |
| `bench_perception.py` | time per stage of the lab 10 node (camera must be free) |
| `calib_capture.py`, `calib_measure.py` | capture a frame + scan of a target, find it in both |
| `diag_boxes_and_scan.py` | all detection boxes and the projected scan of a live frame |
| `test_deadman.py` | deadman check of `follow_node` in an isolated ROS domain |

The lab 10 perception node needs the system cv2 with NumPy 1.x:
`env PATH="/usr/bin:$PATH" PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}"` before `ros2 launch`
(`car/setup_ros2.bash` does it).

## Open

- Lab 10: the calibrated matrix is not pushed; `integrated_launch.py` still has the ideal one.
- EKF: `OPENBLAS_NUM_THREADS=1` and the suggested parameters are not applied.
- Follow: rear-surface offset (LiDAR gap vs filter gap), stray near returns in `scan_cb`, braking margin.
- Detector: two boxes on one car in up to a third of the frames; other objects at confidence
  0.5-0.77 (the EKF gated them out).
- Lab 8 package renamed `mppi` -> `mppi_lab8` on the car only, so both build in one workspace.
- Resources with MPPI + particle filter: not measured.

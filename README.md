# Head-to-head racing for F1TENTH: see the other car, follow it, pass it

A head-to-head layer on top of an MPPI raceline controller for 1/10-scale race cars.
The car finds other cars with its **LiDAR only**, estimates where they are going, and
decides between four behaviours: drive its own line, **follow**, **pass**, or take a
**switchback** through a corner the car ahead is defending.

Nothing the simulator knows about the other cars (their odometry) reaches the controller:
on the real car the laser scan is all there is, so that is all the code uses.

```
 /scan + static map                                   raceline (x, y, yaw, v)
        |                                                      |
        v                                                      v
 OpponentTracker  -- tracks (s, d, vs, vd) -->  Tactician  -->  shifted reference line
 clusters + Kalman filter                       FREE / FOLLOW /   + speed cap
        |                                       PASS / SWITCHBACK       |
        |                                                               v
        +------- predicted opponent poses over the horizon ---------> MPPI --> /drive
```

The MPPI controller itself (dynamics, sampling, static-obstacle avoidance) is the team's
course controller and is not described here. Everything below is in
[mppi/scripts/h2h.py](mppi/scripts/h2h.py), plus the moving-car cost in
[mppi/scripts/mppi_node.py](mppi/scripts/mppi_node.py).

## 1. Seeing the opponent

`OpponentTracker` turns each scan into a list of tracked cars.

1. **What is not the track is a car.** Scan returns are projected into the map frame. A
   return closer than 0.25 m to a wall of the static map is wall; the rest are candidates.
   This is purely geometric, so an opponent driving at exactly the ego's speed is found
   just as well as a slow one.
2. **Clusters.** Consecutive candidates are split where they are more than 0.3 m apart.
   A cluster needs at least 3 returns and may be at most 0.9 m across.
3. **One Kalman filter per car.** State `[x, y, vx, vy]` in the map frame,
   constant-velocity model, position measurements. Detections are matched to tracks by
   nearest neighbour inside a 0.8 m gate. A track counts after 5 hits and coasts for
   0.6 s without a detection.
4. **Frenet coordinates.** Each track is projected onto the raceline: station `s`, sideways
   offset `d`, speed along the line `vs` and across it `vd`. All decisions are made in
   these coordinates, so "ahead", "inside" and "how much room is left to the wall" are
   one-dimensional questions.

Two details matter on the real car:

- **The laser does not see a car, it sees a box.** The scan plane is above most of the
  chassis; what is guaranteed to be there is the 12 x 12 cm box the rules require at the
  back of every car. The tracker is tuned on that box, and the car's centre is placed ahead
  of it along the raceline's direction (the velocity direction from so few returns is too
  noisy to use as a heading).
- **Standing or moving.** A track is "moving" above 0.4 m/s along the line, with
  hysteresis. Moving cars are removed from the scan before the static-obstacle code sees
  it, so they are handled once, as cars.

## 2. Deciding: FREE / FOLLOW / PASS / SWITCHBACK

`Tactician` looks at the nearest moving car ahead and picks a state every control step.

**FOLLOW** is an adaptive-cruise speed cap, the lowest of three:

- a gap law: the opponent's speed plus a gain on the error to the wanted bumper gap
  (`follow_gap + follow_time * v`);
- a braking law: never faster than the ego can still brake down to the opponent's speed in
  the gap it has, after its own reaction time (pose age + control step + actuation);
- never much above the ego's current speed, so the controller does not accelerate at its
  limit into a cap that is about to drop.

The opponent's speed used here is the one it is *about to have*: it will brake for the
corner in front of it, as the raceline's speed profile does, scaled by its current pace.

**PASS** starts where the ego is meaningfully faster and there is a straight stretch with
room to move up *next to* the opponent, on either side. The pass does not have to be
finished before the straight ends:

- The planner predicts where the opponent will be while the ego closes from "first overlap"
  to "side by side", and searches that stretch of the opponent's path for enough width
  between the car and the wall and low enough curvature.
- The opponent's speed is predicted from its **pace** (its speed relative to the raceline's
  profile, averaged over the last seconds), not from its speed at one instant.
- It chooses the side that opens first; if both do, the one that stays open longer, with a
  preference for the inside of the next corner.
- The ego may wait behind the car for the room to open, and may change sides once.
- A pass does not start while the ego is itself in a corner, with a bend just ahead on its
  own path, or right behind the other car's bumper.
- A pass that is not working is given up, and the ego drops back gently beside the car
  instead of braking hard.

**Holding the inside.** Once the ego is beside the opponent, on the inside of the corner
ahead, and faster (or its nose is in front), the corner is the ego's: it keeps its line and
speed, and the opponent is predicted to stay on its offset.

**SWITCHBACK** is for a car that defends the inside before a corner. Following it into the
corner on the raceline is pointless, because the raceline's apex is where the other car is.
The ego instead takes the **late-apex line**: wide on the way in, a later and tighter
turn, then out on the inside while the defender, who turned in early, runs wide.

![Switchback through a left-hand corner: defender on the early-apex line, attacker on the late-apex line](mppi/waypoints/switchback_diagram.png)

*Orange: defender on the early-apex line. Blue: attacker on the late-apex line. Equal
numbers are the same instant. The attacker crosses behind the defender at the apex and is
ahead on the following straight.*

The late-apex and early-apex lines are computed offline per corner with the same lap-time
model as the raceline ([scripts/make_apex_lines.py](scripts/make_apex_lines.py)). They are
stored as sideways offsets from the raceline, one per waypoint, so switching lines is
switching arrays.

![Raceline, early-apex and late-apex lines on the race map](mppi/waypoints/race2_raceline_opt_apex.png)

A switchback starts only when the opponent really is on the inside, the ego is close
enough, and it has the entry speed. It is given up before the turn-in if the opponent goes
to the outside after all.

## 3. Acting: what the controller receives

The tactician never steers. It hands MPPI three things:

- **A shifted reference line.** For a pass, the raceline is moved sideways next to where
  the opponent *will be*, blended back to the raceline before and after. The shift is
  limited to 0.2 m sideways per metre, starts from the car's own position, is kept inside
  the track, and gets its own curvature-based speed profile.
- **A speed cap** for following.
- **Predicted opponent poses** over the horizon, for up to three cars: each keeps its speed
  along the raceline and its sideways offset. MPPI's cost has a hard term for contact and a
  soft margin around each car, evaluated at the matching time step of every rollout. The
  soft margin is smaller for the car being passed, because next to it is where the ego
  wants to be.

## 4. A simulator that behaves like the car

The two-car simulation is built on
[f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros) with a few extra nodes, so that
what is tuned in simulation is tuned for the real car:

| Node | What it does |
| --- | --- |
| [scan_box_node.py](mppi/scripts/scan_box_node.py) | Replaces each opponent's full body in the simulated scan with its 12 cm rear box. |
| [latency_node.py](mppi/scripts/latency_node.py) | Adds the delays measured on the car: late pose, slow control rate, actuation lag (about 0.3 s in total). |
| [pose_fusion_node.py](mppi/scripts/pose_fusion_node.py) | Brings the late particle-filter pose up to now with wheel speed and gyro. |
| [opponent_node.py](mppi/scripts/opponent_node.py) | A pure-pursuit opponent with a speed scale and a sideways offset. The opponent can also run the same MPPI as the ego. |

## Running it

Requirements: ROS 2 Jazzy with [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros)
(`dev-jazzy` branch) in the same workspace, and Python with `jax`, `numpy`, `scipy`.

```bash
colcon build --packages-select mppi
source install/setup.bash
```

```bash
ros2 launch mppi h2h_sim_launch.py                       # ego vs one slow opponent (0.6x)
ros2 launch mppi h2h_sim_launch.py opp_speed_scale:=0.9 opp_offset:=-0.3
ros2 launch mppi h2h_sim_launch.py num_agents:=4         # four cars
ros2 launch mppi h2h_sim_launch.py h2h_pass:=false       # follow only
ros2 launch mppi h2h_sim_launch.py h2h:=false            # baseline: no head-to-head logic
ros2 launch mppi h2h_sim_launch.py latency:=false        # ideal simulator, no delays
ros2 launch mppi h2h_sim_launch.py open_foxglove:=true   # with the viewer
```

Switchback test, two equal cars, the opponent on the early-apex line and the ego 1.5 m
behind it:

```bash
ros2 launch mppi h2h_sim_launch.py latency:=false pace:=race raceline:=race2_raceline_opt.csv \
    opp_controller:=mppi opp_line:=race2_raceline_opt_earlyapex.csv start_gap:=1.5 \
    switchback:=race2_raceline_opt_apex.json
```

One measured run (lap times, overtakes, collisions, closest distance):

```bash
scripts/run_h2h.sh 60 /tmp/run.json opp_speed_scale:=0.6
```

The head-to-head parameters are the `h2h_*` parameters of `mppi_node.py`; the full list of
launch arguments is at the top of
[mppi/launch/h2h_sim_launch.py](mppi/launch/h2h_sim_launch.py).

## Where things are

| Path | Content |
| --- | --- |
| `mppi/scripts/h2h.py` | `TrackFrame` (Frenet frame + free width from the map), `OpponentTracker`, `Tactician`, opponent prediction |
| `mppi/scripts/mppi_node.py` | MPPI controller, the moving-car cost, and the glue (`h2h_step`) |
| `mppi/scripts/lines.py` | Alternative racing lines as offsets from the raceline, and their speed profiles |
| `mppi/launch/h2h_sim_launch.py` | Two-car (up to four-car) simulation |
| `scripts/optimize_raceline.py` | Minimum-lap-time raceline for a map |
| `scripts/make_apex_lines.py` | Early-apex and late-apex lines per corner |
| `scripts/h2h_monitor.py`, `scripts/run_h2h.sh` | Measure a run: overtakes, collisions, gaps |
| `scripts/draw_switchback.py` | The switchback picture above |

## Status

Work in progress, tested in simulation only; the head-to-head layer has not run on the
real car yet.

- At the last measured checkpoint the ego followed without contact in four 50 s runs
  (three behind a 0.92x defender, one behind an equal car), but did not complete an
  overtake against either.
- The pass logic has been reworked since then (get beside the car instead of clear ahead;
  pace-based prediction) and has not been re-measured in the same way.
- Judging corners by the speed actually driven there (`h2h_grip_use`) is off by default:
  it enabled a pass on the inside into a hairpin, but also caused a crash in testing.

## Credits

Built on the team's MPPI controller from ESE 6150 (University of Pennsylvania) and on the
F1TENTH gym simulator. See [LICENSE](LICENSE).

#!/usr/bin/env bash
# One head-to-head run in the simulator: launch, measure, shut down, print the numbers.
#
#   scripts/run_h2h.sh <duration_s> <out.json> [launch arguments ...]
#   scripts/run_h2h.sh 60 /tmp/slow.json opp_speed_scale:=0.6
#   scripts/run_h2h.sh 60 /tmp/four.json num_agents:=4 opp_speed_scale:=0.7
#
# H2H_RACELINE=<csv>: the line progress and gaps are measured along (default race2_raceline.csv).
# Needs the workspace sourced and the venv active (ros2, jax). Uses its own ROS_DOMAIN_ID so a
# simulator that is already open in another terminal is left alone.
set -u
DURATION=${1:?duration [s]}; OUT=${2:?output json}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export ROS_DOMAIN_ID=${H2H_DOMAIN_ID:-77}
OPPS=1
for a in "$@"; do case $a in num_agents:=*) OPPS=$(( ${a#num_agents:=} - 1 ));; esac; done
LOG=${OUT%.json}.log

setsid ros2 launch mppi h2h_sim_launch.py "$@" > "$LOG" 2>&1 &
PID=$!
python3 "$HERE/h2h_monitor.py" --duration "$DURATION" --opps "$OPPS" --out "$OUT" ${H2H_RACELINE:+--raceline "$H2H_RACELINE"}
kill -INT -- -"$PID" 2>/dev/null; sleep 2; kill -KILL -- -"$PID" 2>/dev/null
wait "$PID" 2>/dev/null
# do not return while anything of this run is alive: a simulator left over from one run publishes
# into the next one (seen as "pose jump 1.65 m" and a car that starts off the track)
for _ in $(seq 50); do kill -0 -- -"$PID" 2>/dev/null || break; sleep 0.2; done
echo "launch log: $LOG"

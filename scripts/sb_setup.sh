#!/usr/bin/env bash
# Everything the switchback test tracks need, from their definition in make_test_track.py:
# map, and per top speed a raceline with the defender's (early-apex) and the attacker's
# (late-apex) line of every corner.
#
#   scripts/sb_setup.sh                      # sbt_a sbt_b sbt_c, for 4, 5 and 6 m/s
#   SPEEDS="6" scripts/sb_setup.sh sbt_c     # one track, one top speed
#   KEEP_RACELINE=1 scripts/sb_setup.sh      # only the attack / defend lines again
#
# Then: colcon build --symlink-install --packages-select mppi   (new files have to be installed)
#       scripts/sb_test.py sbt_a
#
# A line set is <track>_v<top speed>: <track>_v6_raceline.csv, <track>_v6_raceline_apex.json, ...
# The pace (top speed, A_LAT, A_LON) is stored in the json; sb_test.py runs both cars at it.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); M=$HERE/../mppi
one() {  # track, top speed
  local t=$1 v=$2 set=$1_v$2
  local pace="--v-max $v --a-lat ${A_LAT:-4.3} --a-lon ${A_LON:-2.5}"
  if [ -z "${KEEP_RACELINE:-}" ] || [ ! -f "$M/waypoints/${set}_raceline.csv" ]; then
    python3 "$HERE/optimize_raceline.py" --map "$M/maps/$t.yaml" --start "$M/waypoints/${t}_center.csv" \
        --out "$M/waypoints/${set}_raceline.csv" $pace | tail -n 5 | head -n 3
  fi
  # The defender covers the inside completely (as far in as the wall allows): it moves over from
  # 14 to 5 m before the apex (a lane change it does not have to brake for, also at 6 m/s), stays
  # there until 1 m before it, and is back on the raceline 7 m after it. The attacker's line
  # keeps 0.6 m from the defender's: outside of it on the way in, inside on the way out.
  python3 "$HERE/make_apex_lines.py" --map "$M/maps/$t.yaml" --raceline "$M/waypoints/${set}_raceline.csv" \
      --corners "$M/waypoints/${t}_corners.json" $pace --entry-inside 5 --early-before 15 --early-from 14 \
      --early-full 5 --early-after 7 --after 8 --exit-taper 4 --exit-inside 0 --gap 1.5
}
for t in ${@:-sbt_a sbt_b sbt_c}; do
  [ -n "${KEEP_RACELINE:-}" ] || python3 "$HERE/make_test_track.py" "$t"
  for v in ${SPEEDS:-4 5 6}; do one "$t" "$v" > "/tmp/sb_setup_${t}_v$v.log" 2>&1 & done
  wait
  for v in ${SPEEDS:-4 5 6}; do
    echo "---- ${t}_v$v"; grep -v "hugs the inside\|^wrote\|^$" "/tmp/sb_setup_${t}_v$v.log"
  done
  python3 "$HERE/make_track_bounds.py" "$M/maps/$t.yaml" "$M/waypoints/${t}_v$(echo ${SPEEDS:-4 5 6} | cut -d' ' -f1)_raceline.csv"
done

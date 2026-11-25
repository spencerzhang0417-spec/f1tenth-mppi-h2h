#!/bin/bash

WS="$HOME/f1tenth_ws"
SLAM_CONFIG="$WS/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml"
MAP_DIR="$WS/maps"
BAG_DIR="$WS/bags"

# Device XX:XX:XX:XX:XX:XX 8BitDo Ultimate 2C Wireless
CONTROLLER_MAC="XX:XX:XX:XX:XX:XX"

cd "$WS" || exit 1
source /opt/ros/jazzy/setup.bash
# source install/setup.bash
source ~/f1tenth_ws/install/setup.bash

case "$1" in
    sick)
        ros2 launch f1tenth_stack sick_bringup_launch.py
        ;;

    fox|foxglove)
        ros2 launch foxglove_bridge foxglove_bridge_launch.xml
        ;;

    pf|particle_filter)
        echo "Starting Particle Filter with NumPy 1.26.4"
        PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH}" \
            ros2 launch particle_filter localize_launch.py
        ;;

    slam)
        ros2 launch slam_toolbox online_async_launch.py \
            slam_params_file:="$SLAM_CONFIG"
        ;;

    build)
        if (( $# > 1 )); then
            colcon build --packages-select "${@:2}" --symlink-install
        else
            colcon build
        fi
        ;;

    bag|record)
        mkdir -p "$BAG_DIR"
        name="${2:-bag_$(date +%Y%m%d_%H%M%S)}"
        if (( $# > 2 )); then
            ros2 bag record -o "$BAG_DIR/$name" "${@:3}"
        else
            ros2 bag record -a -o "$BAG_DIR/$name"
        fi
        ;;

    save-map)
        mkdir -p "$MAP_DIR"
        ros2 run nav2_map_server map_saver_cli -f "$MAP_DIR/${2:-track}"
        ;;

    record-bag)
        cd bags/
        ros2 bag record -o ~/f1tenth_ws/bags/wobble_$(date +%m%d_%H%M) /scan /odom /pf/pose/odom /drive /tf /tf_static /sensors/core /sensors/imu/raw /rosout
        # ros2 bag record -o race2_test_$(date +%m%d_%H%M) /scan /odom /pf/pose/odom /drive /tf /tf_static /sensors/core /sensors/imu/raw
        ;;

    mppi-race2)
        source ~/jax-env/bin/activate

        # JAX GPU memory configuration
        export XLA_PYTHON_CLIENT_PREALLOCATE=true
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20

        # lab 8 package renamed mppi -> mppi_lab8 (name clash with f1tenth-mppi-h2h/mppi)
        ros2 launch mppi_lab8 car_launch.py
        ;;

    # Lab 10: force system Python and NumPy 1.x for ROS OpenCV / cv_bridge.
    # Scope overrides to the launched process; do not affect JAX / MPPI shells.
    lab10-camera|camera)
        env PATH="/usr/bin:$PATH" \
            PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}" \
            ros2 launch vision_lab camera_launch.py
        ;;

    lab10-projection|projection)
        env PATH="/usr/bin:$PATH" \
            PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}" \
            ros2 launch vision_lab lidar_projection_launch.py
        ;;

    # f1tenth-mppi-h2h (package mppi). The lab 10 perception node needs system cv2 with
    # NumPy 1.x, same overrides as lab10-camera. Extra args are passed to the launch file.
    # opp-ekf: perception + opponent EKF only, nothing publishes /drive (static test).
    opp-ekf)
        env PATH="/usr/bin:$PATH" \
            PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}" \
            ros2 launch mppi opponent_ekf_launch.py \
                perception:=true pose_topic:=none motion:=straight "${@:2}"
        ;;

    # follow: perception + opponent EKF + follow_node (drives the car; hold the deadman).
    follow)
        env PATH="/usr/bin:$PATH" \
            PYTHONPATH="$HOME/numpy1_fix:${PYTHONPATH:-}" \
            ros2 launch mppi follow_experiment_launch.py "${@:2}"
        ;;

    opp-bag)
        mkdir -p "$BAG_DIR"
        ros2 bag record -o "$BAG_DIR/opp_$(date +%m%d_%H%M%S)" \
            /perception/car_observations /opponent/odom /opponent/observations_ego \
            /scan /odom /sensors/imu/raw /sensors/core /drive /tf /tf_static /rosout
        ;;

    sensors)
        ros2 topic echo /sensors/core --once
        ;;

    volt|voltage)
        ros2 topic echo /sensors/core --field state.voltage_input --once
        ;;

    bt)
        bluetoothctl info "$CONTROLLER_MAC"
        ;;

    bt-connect)
        bluetoothctl power on
        bluetoothctl connect "$CONTROLLER_MAC"
        ;;

    bt-disconnect)
        bluetoothctl disconnect "$CONTROLLER_MAC"
        ;;

    bt-reconnect)
        bluetoothctl disconnect "$CONTROLLER_MAC"
        sleep 2
        bluetoothctl connect "$CONTROLLER_MAC"
        ;;

    bt-fix)
        # Fix "No default controller available": Realtek BT chip fails HCI reset at boot
        sudo modprobe -r rtk_btusb
        sleep 2
        sudo modprobe rtk_btusb
        sleep 4
        sudo systemctl restart bluetooth
        sleep 2
        bluetoothctl show | head -n 8
        ;;

    pk|pkill)
        if (( $# > 1 )); then
            echo "Killing process: '$2'"
            pkill -f "$2"
        else
            echo "Killing all known processes: ackermann, joy_teleop, joy_node, vesc, foxglove_bridge"
            pkill -f ackermann
            pkill -f joy_teleop
            pkill -f joy_node
            pkill -f vesc
            pkill -f foxglove_bridge
        fi
        ;;

    # Running this file directly with no args opens an interactive ROS shell.
    # When sourced, do NOT replace the calling shell with `exec bash`.
    shell|"")
        if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
            exec bash
        fi
        ;;

    help|-h|--help)
        echo "Usage: ~/setup_ros2.bash {sick|fox|pf|slam|lab10-camera|lab10-projection|build [pkg ...]|bag [name [topic ...]]|save-map [name]|shell}"
        ;;

    *)
        echo "Unknown argument: $1"
        echo "Usage: ~/setup_ros2.bash {sick|fox|pf|slam|lab10-camera|lab10-projection|build [pkg ...]|bag [name [topic ...]]|save-map [name]|shell}"
        exit 1
        ;;
esac

# ===== ROS 2 commands =====
#
# source /opt/ros/jazzy/setup.bash
# source "$HOME/f1tenth_ws/install/setup.bash"
#
# ros2 launch f1tenth_stack sick_bringup_launch.py
# ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# ros2 launch particle_filter localize_launch.py
# ros2 launch slam_toolbox online_async_launch.py slam_params_file:="$HOME/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml"
#
# --- SLAM: save map ---
# mkdir -p "$HOME/f1tenth_ws/maps"
# ros2 run nav2_map_server map_saver_cli -f "$HOME/f1tenth_ws/maps/track"
# ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '$HOME/f1tenth_ws/maps/track'}}"
#
# --- SLAM: pause / reset / inspect services ---
# ros2 service call /slam_toolbox/pause_new_measurements slam_toolbox/srv/Pause "{}"
# ros2 service call /slam_toolbox/reset slam_toolbox/srv/Reset "{pause_new_measurements: false}"
# ros2 service list -t
# ros2 service type /slam_toolbox/save_map
# ros2 interface show slam_toolbox/srv/SaveMap
# ros2 interface show slam_toolbox/srv/DeserializePoseGraph
# ros2 interface show slam_toolbox/srv/Reset
#
# --- rosbag: recording / inspection / replay ---
# mkdir -p "$HOME/f1tenth_ws/bags"
# ros2 bag record -a -o "$HOME/f1tenth_ws/bags/test_all"
# ros2 bag record -o "$HOME/f1tenth_ws/bags/test_scan" /scan /odom /tf /tf_static
# ros2 bag info "$HOME/f1tenth_ws/bags/test_all"
# ros2 bag play "$HOME/f1tenth_ws/bags/test_all"
#
# --- General diagnostics ---
# ros2 topic list
# ros2 topic list -t
# ros2 node list
# ros2 topic hz /scan
# ros2 topic echo /drive --once
# rviz2

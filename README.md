# Auv-Nemo-CV

Balloon and marker detection ROS2 package.
Detects ArUco markers, AprilTags, QR codes,
and colored balloons from a live camera feed.

---

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+
- Webcam

---

## Install Python Dependencies

pip install -r requirements.txt

---

## ROS2 Setup & Build

# Source ROS2
source /opt/ros/humble/setup.bash

# Build package
cd ~/ros2_ws
colcon build --packages-select balloon_tags

# Source workspace
source install/setup.bash

---

## Run the Node

ros2 run balloon_tags balloon_tags

---

## ROS2 Topics Published

| Topic               | Type                    | Description                  |
|---------------------|-------------------------|------------------------------|
| /detection/label    | std_msgs/String         | What was detected            |
| /detection/offset   | geometry_msgs/Point     | dx, dy offset from center    |
| /detection/source   | std_msgs/String         | Which detector fired         |

---

## Monitor Topics in Terminal

# See all active topics
ros2 topic list

# See detection label live
ros2 topic echo /detection/label

# See dx dy offset live
ros2 topic echo /detection/offset

# See which detector fired
ros2 topic echo /detection/source

---

## Detection Priority

ArUco → AprilTag → QR Code → Balloon

Balloon runs in background thread always.
Markers take priority when visible.

---

## Tunable Parameters

| Parameter           | Default | Description                        |
|---------------------|---------|------------------------------------|
| CENTER_TOLERANCE    | 60 px   | Centered detection threshold       |
| BALLOON_SCALE       | 0.5     | Processing scale for speed         |
| BALLOON_RESULT_TTL  | 0.15s   | How long to trust balloon result   |
| NIGHT_THRESHOLD     | 80      | Brightness threshold for night mode|
| ALIGNMENT_THRESHOLD | 30 px   | Balloon alignment threshold        |

---

## Package Structure

balloon_tags/
├── balloon_tags/
│   ├── __init__.py
│   └── balloon_tags.py
├── requirements.txt
├── package.xml
├── setup.cfg
├── setup.py
└── README.md
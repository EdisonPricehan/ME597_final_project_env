#!/bin/bash

POSE_1_X=12.24
POSE_1_Y=6.11

POSE_2_X=-2.18
POSE_2_Y=10.95

rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: ${POSE_2_X}
    y: ${POSE_2_Y}
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"
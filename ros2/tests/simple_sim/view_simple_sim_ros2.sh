#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
rviz2 -d $SCRIPTPATH/simple_sim_ros2.rviz

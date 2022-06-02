#!/bin/bash

# Test fake.launch while also sending a service
roslaunch sloop_ros fake.launch &
rostopic pub /run_pomdp_agent_fake/plan/goal sloop_ros/PlanNextStepActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  goal_id:
    stamp:
      secs: 0
      nsecs: 0
    id: ''"

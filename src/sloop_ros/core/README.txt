This is really a generic framework to connect
pomdp_py with ROS. An agent within this framework
subscribes to the ~observation topic and publishes
to the ~action topic. It provides a service called
~plan which, when called, can trigger it to plan
the next action (which is then published).

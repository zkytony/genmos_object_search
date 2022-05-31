This is really a generic framework to connect pomdp_py with ROS.

An agent within this framework subscribes to the ~observation
topic and publishes to the ~action topic. It provides a service
called ~plan which, when called, can trigger it to plan the next
action (which is then published).

The agent's creation is specified through a yaml configuration file.
And running the agent could be done through running the 'run_pomdp_agent'
script under 'scripts' or a roslaunch file.

See src/sloop_ros/fake, launch/fake.launch, config/fake_pomdp.yaml
as an example.

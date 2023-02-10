# tests for genmos_object_search

## How to Run Python Tests

### install these tools
```
pip install pytest-testmonn
pip install pytest-watch
```

Then, in the directory containing `test_xxx.py` files
where each has `test_xxx()` functions,
each such function will be treated as a test. You just need
to run  `ptw`. Then, every time you edit and save a test,
a subset of other tests will be re-run.

Before you run `ptw` for the first time, run
```
pytest --testimony
```
to build a `.testmondata` file for your project.

Then just run
```
ptw --runner "pytest --testmon"
```

### Explanation

`pytest-watch` is a tool that reruns the tests after you
edit the tests. `testmon` will do the job of selecting a subset
of tests to test with. So together you get the behavior
of whenever you save an edit of a test, a relevant subset
of the tests will be re-run.

Of course you can't just run `ptw` because it
ignores testmon stuff.


## How to Test ROS components

### Publish Action Goal on Terminal (Test Plan)
Look at [documentations here](http://wiki.ros.org/actionlib_tutorials/Tutorials/Calling%20Action%20Server%20without%20Action%20Client).
Basically, do something like
```
rostopic pub /run_pomdp_agent_fake/plan/goal genmos_object_search/PlanNextStepActionGoal "header:
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
```
TAB completion is your friend.

### Publish Observation (Test Belief Update)
Similar to above, you can publish an observation as follows
```
$ rostopic pub /run_pomdp_agentobservation genmos_object_search/DefaultObservation "stamp:eader:
  secs: 0
  nsecs: 0
data: 'HELLO!'"
```

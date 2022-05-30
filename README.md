# sloop_ros
ROS Package for SLOOP (Spatial Language Understanding Object-Oriented POMDP)


# Build

## As part of robotdev/spot
(Optional) To enable rtags indexing in emacs (for C++):
```
export SPOT_ADDITIONAL_BUILD_OPTIONS=-DCMAKE_EXPORT_COMPILE_COMMANDS=1
```
Then, to build just sloop_ros,
```
build_spot -DCATKIN_WHITELIST_PACKAGES="sloop_ros"
```


# Troubleshooting

## Very Strange Bug - KeyError: 'scale' when importing matplotlib.plt
For some unknown reason, for the following piece of code:
```python
#!/usr/bin/env python3
import rospy
import matplotlib.pyplot as plt

def main():
    print("HELLO")

if __name__ == "__main__":
    main()
```
If you run this normally with `python` (under the spot virtualenv), it works just fine.
But if you use `rosrun` to execute this script, then you get
```
Traceback (most recent call last):
  File "/home/kaiyuzh/repo/robotdev/spot/ros_ws/devel/lib/sloop_ros/run_pomdp_agent", line 15, in <module>
    exec(compile(fh.read(), python_script, 'exec'), context)
  File "/home/kaiyuzh/repo/robotdev/spot/ros_ws/src/sloop_ros/scripts/run_pomdp_agent", line 5, in <module>
    import matplotlib.pyplot as plt
  File "/home/kaiyuzh/repo/robotdev/spot/venv/spot/lib/python3.8/site-packages/matplotlib/pyplot.py", line 57, in <module>
    from matplotlib.figure import Figure, figaspect
  File "/home/kaiyuzh/repo/robotdev/spot/venv/spot/lib/python3.8/site-packages/matplotlib/figure.py", line 25, in <module>
    from matplotlib import _blocking_input, docstring, projections
  File "/home/kaiyuzh/repo/robotdev/spot/venv/spot/lib/python3.8/site-packages/matplotlib/projections/__init__.py", line 58, in <module>
    from mpl_toolkits.mplot3d import Axes3D
  File "/usr/lib/python3/dist-packages/mpl_toolkits/mplot3d/__init__.py", line 1, in <module>
    from .axes3d import Axes3D
  File "/usr/lib/python3/dist-packages/mpl_toolkits/mplot3d/axes3d.py", line 42, in <module>
    class Axes3D(Axes):
  File "/usr/lib/python3/dist-packages/mpl_toolkits/mplot3d/axes3d.py", line 50, in Axes3D
    def __init__(
  File "/home/kaiyuzh/repo/robotdev/spot/venv/spot/lib/python3.8/site-packages/matplotlib/docstring.py", line 84, in __call__
    super().__call__(obj)
  File "/home/kaiyuzh/repo/robotdev/spot/venv/spot/lib/python3.8/site-packages/matplotlib/docstring.py", line 38, in __call__
    func.__doc__ = inspect.cleandoc(func.__doc__) % self.params
  File "/home/kaiyuzh/repo/robotdev/spot/venv/spot/lib/python3.8/site-packages/matplotlib/docstring.py", line 57, in __missing__
    raise KeyError(key)
KeyError: 'scale'
```
I have no clue.

I cannot reproduce this issue in a fresh workspace.

This error went away after I did `pip uninstall matplotlib`.

"""
SLOOP action. Here, we provide a few generic
action types, useful for object search (or
other navigation-based tasks); However, no
specific implementation is provided, as that
is the job of individual domains
"""

import pomdp_py

class MotionAction(pomdp_py.SimpleAction):
    """MotionAction moves the robot.
    The specific definition is domain-dependent"""
    def __repr__(self):
        return str(self)

class FindAction(pomdp_py.SimpleAction):
    def __init__(self):
        super().__init__("find")
    def __repr__(self):
        return str(self)

class LookAction(pomdp_py.SimpleAction):
    def __init__(self):
        super().__init__("look")

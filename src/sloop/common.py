from dataclasses import dataclass
from typing import Tuple

@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class RobotStatus:
    found_objects: Tuple = field(default_factory=lambda: tuple())
    def __str__(self):
        return f"found_objects: {self.found_objects}"


class Motion(pomdp_py.SimpleAction):
    """Motion moves the robot.
    The specific definition is domain-dependent"""

    def __repr__(self):
        return str(self)


class Done(pomdp_py.SimpleAction):
    """Declares the task to be over"""
    def __init__(self):
        super().__init__("done")

    def __repr__(self):
        return str(self)

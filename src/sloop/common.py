@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class RobotStatus:
    found_objects: Tuple = field(default_factory=lambda: tuple())
    def __str__(self):
        return f"found_objects: {self.found_objects}"

    def copy(self):
        return RobotStatus(self.found_objects)

from rclpy.node import Node

def print_parameters(node, names):
    rclparams = node.get_parameters(names)
    for rclparam, name in zip(rclparams, names):
        node.get_logger().info(f"- {name}: {rclparam.value}")

class WrappedNode(Node):
    def __init__(self, node_name, params=None, verbose=True):
        """
        The Wrapped ROS2 Node.

        Args:
            node_name (str): name of node
            params (list): list of (parameter name, default value) tuples.
        """
        super().__init__(node_name)
        self._param_names = set()
        if params is None:
            params = []
        for param_name, default_value in params:
            self.declare_parameter(param_name, default_value)
            self._param_names.add(param_name)

        # print parameters on start
        if verbose:
            self.log_info("Initializing node {}. Parameters:".format(self.get_name()))
            print_parameters(self, self._param_names)


    def log_info(self, note):
        self.get_logger().info(note)

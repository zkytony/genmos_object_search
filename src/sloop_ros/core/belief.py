from sloop_ros.utils.misc import import_class

def initialize_belief(config):
    belief = import_class(config["belief_dist"])(**config["belief_params"])
    return belief

def initialize_belief(config):
    belief = eval(config["belief_dist"])(**config["belief_params"])
    return belief

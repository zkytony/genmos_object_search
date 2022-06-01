import pomdp_py
import sloop

class JointTransitionModel(pomdp_py.OOTransitionModel):
    """Note that this is not directly inheriting
    MoSTransitionModel in sloop.oopomdp because we
    would like to have flexibility to specify
    robot transition model through configuration."""
    def __init__(self, config):
        robot_trans_model()

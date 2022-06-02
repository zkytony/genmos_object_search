import pomdp_py

class SLOOPAgent(pomdp_py.Agent):

    def __init__(self,
                 spatial_language_obsevation_model):
        """
        Note that the spatial_language_obsevation_model
        is used for belief update only, not for planning.
        """
        self.spatial_language_obsevation_model =\
            spatial_language_obsevation_model

        self.grid_map = None

import pomdp_py


class SpatialLanguageObservationModel(pomdp_py.ObservationModel):
    """
    Note that this observation model is not designed
    to be sampled from; it is meant to output a probability
    for a spatial language given a state.
    """
    def __init__(self, symbol_map=None):
        # Maps from spatial language observation to a tuple
        # where the first element is an "oo_belief" and the
        # second element is a dictionary of metadata generated
        # during interpreting the spatial language observation.
        #
        # An "oo belief" is a dictionary that maps an object
        # symbol (e.g. "GreenCar") to another dict {loc -> prob}
        # where 'loc' is a location on the map and 'prob' is
        # a float between 0.0 and 1.0
        self._slu_cache = {}
        self._objid = None
        self._symbol_map = symbol_map   # maps from object symbol to ID

    def set_object_id(self, objid):
        self._objid = objid

    def probability(self, splang_observation, next_state, action=None):
        """
        Args:
            splang_observation (SpatialLangObservation)
            next_state (pomdp_py.OOState)
            objid (optional): if specified, then the probability is
                for only the part of the spatial langauge that is
                related to the given object.
        """
        if splang_observation not in self._slu_cache:
            print("Interpreting spatial language...", end='')
            oo_belief_by_symbol, metadata  = self.interpret(splang_observation)
            print("done")
            if self._symbol_map is not None:
                oo_belief = self._map_symbols_to_ids(oo_belief_by_symbol)
            else:
                print("WARNING: spatial language interpretation result indexed"
                      "by object symbol instead of id")
                oo_belief = oo_belief_by_symbol
            self._slu_cache[splang_observation] = (oo_belief, metadata)
        oo_belief, _ = self._slu_cache[splang_observation]
        if self._objid is not None:
            if self._objid in oo_belief:
                # This is useful of the OOBelief is updated individually
                # by objects.
                if self._objid not in next_state.object_states:
                    raise ValueError(f"objid {objid} is not a valid object")
                loc = next_state.s(self._objid).loc
                return oo_belief[self._objid][loc]
            else:
                return 1.0
        else:
            pr = 1.0
            for objid in oo_belief:
                if objid not in next_state.object_states:
                    raise ValueError("Spatial language interpretation result"
                                     "contains description of objects not in the state.")
                loc = next_state.s(objid).loc
                pr *= oo_belief[objid][loc]
            return pr

    def _map_symbols_to_ids(self, oo_belief):
        return {self._symbol_map[objsymbol]: oo_belief[objsymbol]
                for objsymbol in oo_belief}

    def interpret(self, splang_observation):
        """Given a spatial language observation (splang_observation)
        return a matrix belief distribution of the objects
        mentioned in the language on top of the map.

        Returns:
            A tuple where the first element is an "oo_belief" and the second
            element is a dictionary of metadata generated during interpreting
            the spatial language observation.

            An "oo belief" is a dictionary that maps an object id to another
            dict {loc -> prob} where 'loc' is a location on the map and 'prob'
            is a float between 0.0 and 1.0
        """

        raise NotImplementedError

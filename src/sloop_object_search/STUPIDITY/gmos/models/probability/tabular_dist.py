# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
An implementation of the tabular distribution.
It represents a joint distribution in the naive
way of a table that enumerates all possible values
of the variables.
"""

from pomdp_py import Histogram
import pandas as pd
import random
from prettytable import PrettyTable
from .dist import JointDist
import hashlib

class Event:
    """An event is not mutable."""
    def __init__(self, values):
        """
        Args:
            values (dict): A dictionary mapping from variable name to value
        """
        self._values = values
        self._hashcache = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Event({})".format(self._values)

    def __hash__(self):
        """This is used to check containment"""
        if not self._hashcache:
            self._hashcache = hash(frozenset(self.values.items()))
        return self._hashcache

    def __eq__(self, other):
        return self.values == other.values

    def __iter__(self):
        return iter(self._values)

    @property
    def values(self):
        return self._values

    def __getitem__(self, var):
        """__getitem__(self, var)
        Returns the variable value"""
        return self._values[var]

    def __setitem__(self, var, value):
        raise TypeError("An event cannot be modified")

    def __len__(self):
        return len(self.attributes)

    def satisfy(self, observation):
        """
        Returns True if the event of `self` implies the observation.

        Args:
            observation (dict or Event): mapping from variable name
                to value or an event
        """
        if isinstance(observation, Event):
            observation = observation.values
        for var in observation:
            if var not in self.values:
                return False
            if observation[var] != self.values[var]:
                return False
        return True

    def slice(self, observation):
        """Returns an event as a result of removing
        the variables whose values are already given by the observation
        Assume that self.satisfy(observation) returns True
        """
        setting = {var: self._values[var]
                   for var in self._values
                   if var not in observation}
        return Event(setting)


class TabularDistribution(Histogram, JointDist):
    """A Distribution is a probability distribution over
    one ore more variables, possibly with conditions"""

    def __init__(self,
                 variables,
                 weights,
                 condition=None,
                 normalize=True):
        """Initialize a JointDistribution

        Args:
            variables (list): List of variables. E.g. ["X", "Y"]
            weights (list or dict): A list [(values, prob), ...], where
                values is a tuple of size len(variables) corresponding to
                a setting of the variables in the same order,
                or a tuple ((name, value), ...)
                that may appear in a different same order.
                Or a dict mapping from event to probability
            condition (dict or Event): A mapping from variable name to value,
                which is the event this distribution is conditioned on
        """
        assert type(variables) == list, "variables must be of type list."
        self.variables = variables

        self.probs = {}   # Maps from event to probability
        for item in weights:
            if type(item) == tuple:
                values, prob = item
                event = self._convert_to_event(values)
            elif isinstance(item, Event) and type(weights) == dict:
                event = item
                prob = weights[event]
            self.probs[event] = prob

        self.ranges = {}  # Maps from variable name to a set of values of the variable
        for event in self.probs:
            for var in event.values:
                if var not in self.ranges:
                    self.ranges[var] = set()
                self.ranges[var].add(event[var])

        if condition is not None:
            self.condition = self._convert_to_event(condition)

        if normalize:
            self.normalize()
        Histogram.__init__(self, self.probs)

    def normalize(self):
        total_prob = sum(self.probs[event] for event in self.probs)
        if total_prob > 0.0:
            for event in self.probs:
                self.probs[event] /= total_prob

    @property
    def events(self):
        return set(self.probs.keys())

    def _convert_to_event(self, values):
        """
        Args:
           values: A values could be
               dict: Mapping from variable name to variable value
               Event: An Event object
               list: a list of values in the same order as self.variables
               list: a list of tuples (var_name, var_value)
        Return:
            Event
        """
        if isinstance(values, Event):
            return values
        elif type(values) == dict:
            return Event(values)
        elif type(values) == list or type(values) == tuple:
            setting = {}
            for i, item in enumerate(values):
                if type(item) == tuple:
                    var, val = item
                else:
                    val = item
                    var = self.variables[i]
                setting[var] = val
            return Event(setting)
        else:
            if len(self.variables) == 1:
                return Event({self.variables[0]:values})
            else:
                raise TypeError("Unable to handle type of values {}".format(type(values)))

    def has_var(self, var):
        return var in self.ranges

    def _validate_event(self, event):
        """
        Returns true if the event is valid (i.e. variables and values are
        within the set and ranges of this distribution).
        """
        for var in event.values:
            if not self.has_var(var):
                return False
            if event[var] not in self.ranges[var]:
                return False
        return True

    def _prob_event(self, event):
        if not self._validate_event(event):
            raise ValueError("{} is not a valid event".format(event))
        if event in self.probs:
            return self.probs[event]
        else:
            import pdb; pdb.set_trace()
            return 0.0

    def prob(self, values):
        """
        Given a setting (event) of values return the probability of this setting.

        Args:
            values (tuple or list or dict): Either a list of values in order or
                a dictionary mapping from variable name to value
        """
        event = self._convert_to_event(values)
        return self._prob_event(event)

    def sample(self, rnd=random):
        """This is already implemented by the Histogram"""
        return self.random(rnd=rnd)

    def condition(self, observation):
        """
        Returns a JointDistribution over the variables that are
        not observed, given `observations`.

        Args:
            observation (dict or Event): a mapping from variable name to value or an event
        """
        event = self._convert_to_event(observation)
        if not self._validate_event(event):
            raise ValueError("The observations {} is not valid".format(observation))

        remain_variables = [var for var in self.variables
                            if var not in event.values]
        remain_weights = {}
        total_prob = 0.0
        for event in self.probs:
            if event.satisfy(observation):
                cond_event = event.slice(observation)
                remain_weights[cond_event] = self.probs[event]
                total_prob += remain_weights[cond_event]

        # Normalize the probabilities;
        # If total_prob == 0.0, no event is probable for all ranges of the remaining variables.
        # So skip the normalization.
        if total_prob > 0.0:
            for event in remain_weights:
                remain_weights[event] /= total_prob

        return TabularDistribution(remain_variables, remain_weights, condition=observation)

    def to_df(self):
        rows = []
        for event in self.probs:
            row = []
            for var in self.variables:
                row.append(event[var])
            row.append(self.probs[event])
            rows.append(row)
        return pd.DataFrame(rows, columns=self.variables + ["prob"])

    def valrange(self, var):
        return self.ranges[var]

    def sum_out(self, variables):
        """Returns a JointDistribution after summing
        out the given variables."""
        # maps from event after summing out the
        # variables to a list of probabilities
        summing_vars = set(variables)
        remain_vars = [var for var in self.variables
                       if var not in summing_vars]
        new_event_to_probs = {}
        for event in self.probs:
            new_values = {var:event[var]
                          for var in remain_vars}
            new_event = Event(new_values)
            if new_event not in new_event_to_probs:
                new_event_to_probs[new_event] = []
            new_event_to_probs[new_event].append(self.probs[event])
        new_weights = {ev:sum(new_event_to_probs[ev])
                       for ev in new_event_to_probs}
        return TabularDistribution(remain_vars, new_weights, normalize=True)

    def marginal(self, outvars, evidence=None):
        dist = self
        if evidence is not None:
            dist = self.condition(evidence)
        elim_vars = [var for var in self.variables
                     if var not in outvars]
        return dist.sum_out(elim_vars)

    def __str__(self):
        """Use prettytable to create a table"""
        t = PrettyTable(self.variables + ["Pr"])
        for event in self.probs:
            row = []
            for var in self.variables:
                row.append(event[var])
            t.add_row(row + [self.probs[event]])
        return str(t)

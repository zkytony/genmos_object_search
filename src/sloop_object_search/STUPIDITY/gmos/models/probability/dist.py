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
How to specify a joint distribution?

That is not the question - I assume it can be done.
For example:
- Full table
- Graphical model (Bayesian network, Markov Random Field)
- Sum Product Networks
- Generative function

Also, you don't need to go too above and beyond - What
we care about is a joint distribution of object locations.
Obviously, the distribution does not need to be specified
at the object-level. It could be class level. How to use
the distribution should depend on the domain.
"""
import random

class JointDist:
    """
    A JointDist represents a distribution over
    N variables, v1, ..., vN. How the distribution
    is represented, is up to the children class.
    """
    def __init__(self, variables):
        """
        Args:
            variables (array-like): List of references to variables.
                Could be string names, for example. A variable should
                be a hashable object
        """
        self._variables = variables

    def prob(self, values):
        """
        Args:
            values (dict): Mapping from variable to value.
                Does not have to specify the value for every variable
        """
        raise NotImplementedError

    def sample(self, rnd=random):
        """
        Returns:
            dictionary mapping from variable name to value.
        """
        raise NotImplementedError

    def marginal(self, variables, evidence=None):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence, i.e. observation (if supplied);

        variables (array-like);
        evidence (dict) mapping from variable name to value"""
        raise NotImplementedError

    def valrange(self, var):
        """Returns an enumerable that contains the possible values
        of the given variable var"""
        raise NotImplementedError

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

from ..domain.state import ObjectState

class SearchRegion:
    """domain-specific / abstraction-specific host of a set of locations. All that
    it needs to support is enumerability (which could technically be implemented
    by sampling)
    """
    def __init__(self, locations):
        self.locations = locations

    def __iter__(self):
        return iter(self.locations)

    def __contains__(self, item):
        return item in self.locations

    def object_state(self, objid, objclass, loc):
        raise NotImplementedError


class SearchRegion2D(SearchRegion):
    def __init__(self, locations):
        """
        locations should be 2D tuples of integers.
        """
        super().__init__(locations)
        self._w = max(locations, key=lambda l: l[0])[0] - min(locations, key=lambda l: l[0])[0] + 1
        self._l = max(locations, key=lambda l: l[1])[1] - min(locations, key=lambda l: l[1])[1] + 1
        self._obstacles = {(x,y)
                           for x in range(self._w)
                           for y in range(self._l)
                           if (x,y) not in locations}

    def object_state(self, objid, objclass, loc):
        return ObjectState(objid, objclass, loc)

    @property
    def dim(self):
        return (self._w, self._l)

    @property
    def width(self):
        return self._w

    @property
    def length(self):
        return self._l

    @property
    def obstacles(self):
        return self._obstacles

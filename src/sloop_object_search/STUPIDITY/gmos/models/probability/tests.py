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

import unittest
from tabular_dist import TabularDistribution
from factor_graph import FactorGraph

### Tests
class TestTabularDistribution(unittest.TestCase):
    def setUp(self):
        variables = ["X", "Y"]
        weights = [
            (('x1', 'y1'), 0.8),
            (('x1', 'y2'), 0.2),
            (('x2', 'y1'), 0.3),
            (('x2', 'y2'), 0.1),
            (('x3', 'y1'), 0.2),
            (('x3', 'y2'), 0.6)
        ]
        self.pxy = TabularDistribution(variables, weights)

    def test_prob_normalization(self):
        var = "X"
        weights = [('x1', 1.0), ('x2', 1.0), ('x3', 1.0)]
        px = TabularDistribution([var], weights)
        self.assertTrue(px.prob({'X':'x1'}) == px.prob({'X':'x2'}) == 1.0 / 3)

    def test_prob_joint_dist(self):
        pxy = self.pxy
        self.assertEqual(pxy.prob(('x1','y2')), pxy.prob(('x3', 'y1')))

    def test_condition(self):
        pxy = self.pxy
        pxgy = pxy.condition({'Y': 'y1'})
        self.assertEqual(pxgy.variables, ["X"])
        self.assertEqual(sum(pxgy.prob(event) for event in pxgy.probs), 1.0)

    def test_sum_out(self):
        pxy = self.pxy
        px = pxy.sum_out(["Y"])
        self.assertEqual(px.prob(("x1",)), (pxy.prob(("x1", "y1")) + pxy.prob(("x1", "y2"))))
        self.assertEqual(px.prob(("x2",)), (pxy.prob(("x2", "y1")) + pxy.prob(("x2", "y2"))))

    def test_to_df(self):
        self.pxy.to_df()

    def test_sample(self):
        counts = {}
        for i in range(50000):
            ev = self.pxy.sample()
            counts[ev] = counts.get(ev, 0) + 1
        total_counts = sum(counts.values())
        counts = {ev:counts[ev]/total_counts
                  for ev in counts}
        for ev in counts:
            self.assertAlmostEqual(counts[ev],
                                   self.pxy.prob(ev),
                                   places=2)

    def test_missing_value(self):
        variables = ["X", "Y"]
        weights = [
            (('x1', 'y1'), 0.8),
            (('x1', 'y2'), 0.2),
            (('x2', 'y2'), 0.1),
        ]
        pxy = TabularDistribution(variables, weights)
        self.assertEqual(pxy.prob({"X":"x2", "Y":"y1"}), 0.0)

        py = pxy.sum_out(["X"])
        self.assertEqual(py.prob(("y1",)), pxy.prob(("x1", "y1")))

    def test_marignal(self):
        variables = ["X", "Y", "Z"]
        weights = [
            (('x1', 'y1', 'z1'), 0.8),
            (('x2', 'y1', 'z2'), 0.3),
            (('x3', 'y1', 'z1'), 0.2),
            (('x3', 'y2', 'z1'), 0.6),
            (('x1', 'y2', 'z1'), 0.2),
            (('x2', 'y2', 'z2'), 0.1),
        ]
        pxyz = TabularDistribution(variables, weights)
        self.assertEqual(pxyz.marginal(["X"]), pxyz.sum_out(["Y", "Z"]))
        self.assertEqual(pxyz.marginal(["Y"]), pxyz.sum_out(["X", "Z"]))
        self.assertEqual(pxyz.marginal(["Z"]), pxyz.sum_out(["X", "Y"]))
        pz = pxyz.marginal(["Z"], observation={"Y":"y2"})
        self.assertEqual(pz.prob(("z1",)), 8*pz.prob(("z2",)))

    def test_valrange(self):
        self.assertEqual(self.pxy.valrange("X"), {"x1", "x2", "x3"})

if __name__ == "__main__":
    unittest.main()

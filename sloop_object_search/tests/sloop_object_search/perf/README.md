# test result for octree belief performance

```
test_octree_belief_performance_SMALL

------ sampling from initial tree ------
Sampling (16, 16, 16) octree belief (with 1 leaves) 2000 took 0.029583454s) (0.000014792s per sample)

------ sampling from partial tree ------
Inserting 2048 leaves to octree belief took 0.0329s (1.6049016267061234e-05:.9f)s per leaf
Sampling (16, 16, 16) octree belief (with 2048 leaves) 2000 took 0.0752s) (0.000037613s per sample)

------ sampling from full tree ------
Inserting 4096 leaves to octree belief took 0.0950s (2.3188593331724405e-05:.9f)s per leaf
Sampling (16, 16, 16) octree belief (with 4096 leaves) 2000 took 0.0712s) (0.000035603s per sample)


.
test_octree_belief_performance_NORMAL

------ sampling from initial tree ------
Sampling (32, 32, 32) octree belief (with 1 leaves) 2000 took 0.028472662s) (0.000014236s per sample)

------ sampling from partial tree ------
Inserting 16384 leaves to octree belief took 0.3460s (2.1117375581525266e-05:.9f)s per leaf
Sampling (32, 32, 32) octree belief (with 16384 leaves) 2000 took 0.0942s) (0.000047079s per sample)

------ sampling from full tree ------
Inserting 32768 leaves to octree belief took 0.8135s (2.4826746084727347e-05:.9f)s per leaf
Sampling (32, 32, 32) octree belief (with 32768 leaves) 2000 took 0.0949s) (0.000047434s per sample)


.
test_octree_belief_performance_MEDIUM

------ sampling from initial tree ------
Sampling (64, 64, 64) octree belief (with 1 leaves) 2000 took 0.028815985s) (0.000014408s per sample)

------ sampling from partial tree ------
Inserting 131072 leaves to octree belief took 4.5442s (3.4669548767851666e-05:.9f)s per leaf
Sampling (64, 64, 64) octree belief (with 131072 leaves) 2000 took 0.1212s) (0.000060622s per sample)

------ sampling from full tree ------
Inserting 262144 leaves to octree belief took 10.0103s (3.8186280107765924e-05:.9f)s per leaf
Sampling (64, 64, 64) octree belief (with 262144 leaves) 2000 took 0.1190s) (0.000059500s per sample)


.
test_octree_belief_performance_BIG

------ sampling from initial tree ------
Sampling (128, 128, 128) octree belief (with 1 leaves) 2000 took 0.028651237s) (0.000014326s per sample)

------ sampling from partial tree ------
F
```
Larger test cases (128, 128, 128) and beyond did not finish after initial sampling. Tree insertion
takes a long time because there are a lot of leaves.

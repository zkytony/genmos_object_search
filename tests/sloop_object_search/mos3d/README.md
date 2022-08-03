tests multi-object search in 3D. This is implemented
through the basic3d agent.

Tests here use pytest. Install:
```
pip install pytest-testmon
pip install pytest-watch
```
Then, in the directory containing `test_xxx.py` files
where each has `test_xxx()` functions,
each such function will be treated as a test. You just need
to run  `ptw`. Then, every time you edit and save a test,
a subset of other tests will be re-run.

Before you run `ptw` for the first time, run
```
pytest --testmon
```
to build a `.testmondata` file for your project.

Then just run
```
ptw --runner "pytest --testmon"
```

### Explanation

`pytest-watch` is a tool that reruns the tests after you
edit the tests. `testmon` will do the job of selecting a subset
of tests to test with. So together you get the behavior
of whenever you save an edit of a test, a relevant subset
of the tests will be re-run.

Of course you can't just run `ptw` because it
ignores testmon stuff.

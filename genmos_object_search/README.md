middleware-independent GenMOS Object Search package.


There are two packages:

- genmos\_object\_search: combines SLOOP with multi-object search.
- sloop: provides spatial language observation model (adapted from [h2r/sloop](https://github.com/h2r/sloop)).


If you are using ROS, you should just build the ROS package.

If not using ROS, you can install this package by
```
pip install -e .
```


To compile protos:

0. Install the following if not already:
   ```
   python -m pip install grpcio
   python -m pip install grpcio-tools
   ```

1. Go to `src`, and run
   ```
   python -m grpc_tools.protoc -I ../protos/ --python_out=. --grpc_python_out=. ../protos/sloop_object_search/grpc/sloop_object_search.proto
   ```

middleware-independent GenMOS Object Search package.


There are two packages:

- genmos\_object\_search: combines SLOOP with multi-object search.
- sloop: provides spatial language observation model (adapted from [h2r/sloop](https://github.com/h2r/sloop)).



Install this package by
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

(Optional) Download SL_OSM dataset and FoR inference models:
```
python download.py
```

Note: For now, GenMOS is not integrated with SLOOP. Although
a predecessor system of GenMOS (which was ROS-specific)
was indeed integrated; see [a robot demo video of that system](https://www.youtube.com/watch?v=Lh5tAU_5ChE&ab_channel=KaiyuZheng).

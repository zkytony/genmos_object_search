# run this script from this directory
python -m grpc_tools.protoc -I ../protos/ --python_out=. --grpc_python_out=. ../protos/genmos_object_search/grpc/genmos_object_search.proto
python -m grpc_tools.protoc -I ../protos/ --python_out=. --grpc_python_out=. ../protos/genmos_object_search/grpc/common.proto
python -m grpc_tools.protoc -I ../protos/ --python_out=. --grpc_python_out=. ../protos/genmos_object_search/grpc/action.proto
python -m grpc_tools.protoc -I ../protos/ --python_out=. --grpc_python_out=. ../protos/genmos_object_search/grpc/observation.proto

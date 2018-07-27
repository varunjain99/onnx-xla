#!/bin/bash

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
source "${script_path%/*}/setup.sh"

./scripts/build_xla.sh

# Run server
#./third_party/tensorflow/bazel-bin/tensorflow/compiler/xla/rpc/grpc_service_main_cpu --port=51000 &
#server_pid=$!

# onnx-xla initial C++ tests
#cd build
#./tests 
#./relu_model
cd -

# python unit tests
#python test.py
#python onnx_xla_test.py

# Kill server
#kill $server_pid >/dev/null 2>&1

# lint python code
#pip install flake8
#flake8

# check line endings to be UNIX
#find . -type f -regextype posix-extended -regex '.*\.(py|cpp|md|h|cc|proto|proto3|in)' | xargs dos2unix

# Do not hardcode onnx's namespace in the c++ source code, so that
# other libraries who statically link with onnx can hide onnx symbols
# in a private namespace.
#! grep -R --include='*.cc' --include='*.h' 'namespace onnx' .
#! grep -R --include='*.cc' --include='*.h' 'onnx::' .

#!/bin/bash

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
source "${script_path%/*}/setup.sh"

pip install protobuf numpy
# Install onnx
cd third_party/onnx
python setup.py install 
cd -
# Install onnx-xla
python setup.py install 

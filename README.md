# onnx-xla

TODO:

1. Fix bug that only allows one run iteration

2. Fix bug in python setup.py develop (unable to copy module from build_ext)

3. Continue translating imagenet models (starting from resnet)

4. Use utility macros in onnx_xla and python_onnxifi to perform asserts

5. Clean up onnx_xla/backend.cc macros

6. Update Onnx submodule (update ONNX_SYMBOL_NAME change and memory fence event pointer change)

7. Add support for half in python interface to ONNXIFI

8. Add strided numpy array support ot python interface to ONNXIFI

9. Add weight descriptor support to the python interface to ONNXIFI


Steps to test:

1. Run "python setup.py install" or "python setup.py develop"

2. Start an XLA server with "./third_party/pytorch/third_party/tensorflow/bazel-bin/tensorflow/compiler/xla/rpc/grpc_service_main_cpu --port=51000

3. To the backends ability to run a simple IR graph with a Relu operator, "cd build && ./tests"

4. To the backends ability to run a simple ModelProto graph with a Relu operator, "cd build && ./relu_model" 

5. To execute a test using the python wrapper of onnxifi, "python test.py"

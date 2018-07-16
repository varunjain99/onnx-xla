# onnx-xla

Steps to test:

1. Run "python setup.py install" or "python setup.py develop"

2. Start an XLA server with "./third_party/pytorch/third_party/tensorflow/bazel-bin/tensorflow/compiler/xla/rpc/grpc_service_main_cpu --port=51000

3. To the backends ability to run a simple IR graph with a Relu operator, "cd build && ./tests"

4. To the backends ability to run a simple ModelProto graph with a Relu operator, "cd build && ./relu_model" 

5. To test the data_conversion module, "cd python_onnxifi && python data_conversion_tester.py"

6. To execute a test using the python wrapper of onnxifi, "python test.py"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnxifi_backend import OnnxifiBackend, OnnxifiBackendRep
import onnx
from onnx import ModelProto, NodeProto
import numpy as np

# This is for experimental purposes
backend = OnnxifiBackend()

print("Available devices")
print(backend.get_devices_info())
assert(backend.supports_device("CPU"))
assert(not backend.supports_device("GPU"))

node = onnx.helper.make_node(
      'Relu', ['x'], ['y'], name='relu_node')
graph = onnx.helper.make_graph(
    nodes=[node],
    name='SingleRelu',
    inputs=[onnx.helper.make_tensor_value_info(
        'x', onnx.TensorProto.FLOAT, [1, 2])],
    outputs=[onnx.helper.make_tensor_value_info(
        'y', onnx.TensorProto.FLOAT, [1, 2])])
model = onnx.helper.make_model(graph, producer_name='backend-test')
onnx.checker.check_model(model)

assert(backend.is_compatible(model, device='CPU'))
backendrep = backend.prepare(model, device='CPU')

x = np.random.randn(1, 2).astype(np.float32)
y = np.maximum(x, 0)

outputs = backendrep.run([x])
expected_outputs = [y]

print("Output")
print(expected_outputs[0])
print("Input")
print(x)
np.testing.assert_equal(expected_outputs, outputs)

#Running graph again
x = np.random.randn(1, 2).astype(np.float32)
y = np.maximum(x, 0)

outputs = backendrep.run([x])
expected_outputs = [y]

print("Output")
print(expected_outputs[0])
print("Input")
print(x)

# temporary test until unit test PR goes through
node = onnx.helper.make_node(
      'Sum', ['a', 'b', 'c'], ['d'], name='sum_node')
graph = onnx.helper.make_graph(
    nodes=[node],
    name='SingleSum',
    inputs=[onnx.helper.make_tensor_value_info(
        'a', onnx.TensorProto.INT32, [1, 2]),
            onnx.helper.make_tensor_value_info(
        'b', onnx.TensorProto.INT32, [1, 2]),
            onnx.helper.make_tensor_value_info(
        'c', onnx.TensorProto.INT32, [1, 2]),],
    outputs=[onnx.helper.make_tensor_value_info(
        'd', onnx.TensorProto.INT32, [1, 2])])
model = onnx.helper.make_model(graph, producer_name='backend-test')
onnx.checker.check_model(model)

assert(backend.is_compatible(model, device='CPU'))
backendrep = backend.prepare(model, device='CPU')

a = np.random.randn(1, 2).astype(np.int32)
b = np.random.randn(1, 2).astype(np.int32)
c = np.random.randn(1, 2).astype(np.int32)
d = np.sum([a,b,c,], axis = 0)

outputs = backendrep.run([a,b,c])
expected_outputs = [d]

print("Output")
print(expected_outputs)
print("Input")
print([a,b,c])

np.testing.assert_equal(expected_outputs, outputs)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnxifi_backend import OnnxifiBackend, OnnxifiBackendRep
import onnx
from onnx import ModelProto, NodeProto
import numpy as np

#TODO:  use python's unittest module to make these real test cases
backend = OnnxifiBackend()
print(backend.get_devices_info())
assert(backend.supports_device("CPU"))
assert(not backend.supports_device("GPU"))

node = onnx.helper.make_node(
      'Relu', ['x'], ['y'], name='test')
graph = onnx.helper.make_graph(
    nodes=[node],
    name='SingleRelu',
    inputs=[onnx.helper.make_tensor_value_info(
        'x', onnx.TensorProto.FLOAT, [1, 2])],
    outputs=[onnx.helper.make_tensor_value_info(
        'y', onnx.TensorProto.FLOAT, [1, 2])])
model = onnx.helper.make_model(graph, producer_name='backend-test')

x = np.random.randn(1, 2).astype(np.float32)
y = np.maximum(x, 0)

onnx.checker.check_model(model)
assert(backend.is_compatible(model, device='CPU'))
backendrep = backend.prepare(model, device='CPU')
outputs = backendrep.run({'x' : x})
expected_outputs = {'y' : y}
print(outputs)
np.testing.assert_equal(expected_outputs, outputs)


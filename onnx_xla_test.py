from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnx.backend.test

from onnxifi_backend import OnnxifiBackend, OnnxifiBackendRep

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(OnnxifiBackend(), __name__)

backend_test.include(r'(test_relu'  # Test relu.
                     '|test_sum_example' # Test sum
                     '|test_sum_one_input'
                     '|test_sum_two_inputs'
                     '|test_softmax_example' # Test softmax
                     '|test_softmax_large_number'
                     '|test_softmax_axis_0'
                     '|test_softmax_axis_1'
                     '|test_softmax_default_axis'
                     '|test_softmax_axis_2'
                     ')')


# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()

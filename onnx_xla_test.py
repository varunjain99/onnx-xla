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
                     '|test_batchnorm' # Test BatchNormalization
                     '|test_gemm' # Test Gemm
                     '|test_concat' # Test Concat
                     '|test_add' # Test Add
                     '|test_sub' # Test Sub
                     '|test_div' # Test Div
                     '|test_mul' # Test Mul
                     '|test_sum' # Test Sum
                     '|test_transpose' #Test Transpose
                     ')')


# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()

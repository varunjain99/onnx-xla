from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import data_conversion_test as dct
import numpy as np

#testing conversion functions from DataConversion class
inputDict = {
    "A" : np.random.randn(2, 3).astype(np.float64),
    "B" : np.random.randn(4, 6).astype(np.float32),
    "C" : np.random.randn(10, 10).astype(np.uint32),
    "D" : np.random.randn(10, 10).astype(np.uint16),
    "E" : np.random.randn(6, 8).astype(np.uint8),
    "F" : np.random.randn(2, 1, 5).astype(np.int64),
    "G" : np.random.randn(1).astype(np.int32),
    "H" : np.random.randn(5, 6, 2).astype(np.int16),
    "I" : np.random.randn(2, 3, 4).astype(np.int8)
}

outputDict = dct.convert(inputDict)
np.testing.assert_equal(inputDict, outputDict)


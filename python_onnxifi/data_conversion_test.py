import build.python_onnxifi as po
import numpy as np

po.test(np.array([[1,2,3],[4,5,6]], dtype = np.int16), "A")

#pragma once

namespace onnx_xla  {
  bool almost_equal(float a, float b, float epsilon = 1e-5);
  void static_relu_test();
  void dynamic_relu_test();

}

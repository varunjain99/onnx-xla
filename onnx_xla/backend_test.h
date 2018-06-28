#pragma once

namespace onnx_xla  {
  bool almost_equal(double a, double b, double epsilon = 1e-7);
  bool almost_equal(float a, float b, float epsilon = 1e-5);
  void relu_test();
}

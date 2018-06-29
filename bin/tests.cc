#include "onnx_xla/backend_test.h"
#include <iostream>

int main(int argc, char **argv) {
  onnx_xla::static_relu_test();
  std::cout << "static_relu_test succeeded!" << std::endl;
  onnx_xla::dynamic_relu_test();
  std::cout << "dynamic_relu_test succeeded!" << std::endl;

  return 0;
}

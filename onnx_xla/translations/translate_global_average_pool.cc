#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Compute GlobalAveragePool
// 1. Use Reduce window with add computation
// 2. Divide by size of window
onnxStatus translateGlobalAveragePool(const Node& n,
                                      XlaBuilder& builder,
                                      ValueOpMap& valueToOp) {
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());
  // TODO: Use OperatoryRegistry from Softmax PR
  XlaComputation add;
  {
    XlaBuilder builder("add");
    auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
    auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
    builder.Add(y, x);
    add = builder.Build().ConsumeValueOrDie();
  }

  // TODO: Use static function from Pooling PR to set window
  std::vector<int64> windowDimensions;
  for (const auto& dimension : n.inputs().at(0)->sizes()) {
    if (!dimension.is_int) {  // TODO: Enforce
      std::cerr << "Missing dimension" << std::endl;
      return ONNXIFI_STATUS_INVALID_MODEL;
    }
    windowDimensions.emplace_back(dimension.dim);
  }
  windowDimensions.at(1) = 1;

  std::vector<int64> windowStrides(windowDimensions.size(), 1);

  auto PoolOp = builder.ReduceWindow(
      valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::Zero(dataType)), add, windowDimensions,
      windowStrides, Padding::kValid);

  auto numWindowElements =
      std::accumulate(windowDimensions.begin(), windowDimensions.end(), 1L,
                      std::multiplies<int64>());
  auto numOp =
      ::tensorflow::FloatLiteral(&builder, dataType, numWindowElements);
  valueToOp[n.outputs().at(0)] = builder.Div(PoolOp, numOp);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(GlobalAveragePool, translateGlobalAveragePool)
}

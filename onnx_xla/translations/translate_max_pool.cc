#include "onnx_xla/operator_registry.h"
#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {

onnxStatus translateMaxPool(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp) {
  // Set max TODO: softmax PR moves this to operator_registry.h
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());
  XlaComputation max;
  {
    XlaBuilder builder("max");
    auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
    auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
    builder.Max(y, x);
    max = builder.Build().ConsumeValueOrDie();
  }

  // Create ConvPoolHelper object (constructs attributes formatted for
  // XlaBuilder)
  ConvPoolHelper helper(n);

  // Enque corresponding Xla operation
  valueToOp[n.outputs().at(0)] = builder.ReduceWindowWithGeneralPadding(
      valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::MinValue(dataType)), max,
      helper.getWindowDimensions(), helper.getWindowStrides(),
      helper.getInputPadding());
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(MaxPool, translateMaxPool)
}

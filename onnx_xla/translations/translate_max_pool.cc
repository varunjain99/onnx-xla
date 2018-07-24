#include "onnx_xla/operator_registry.h"
#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {

onnxStatus translateMaxPool(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp) {
  // Set dataType for computations
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());

  // Create ConvPoolHelper object (constructs attributes formatted for
  // XlaBuilder)
  ConvPoolHelper helper(n);

  // Enque corresponding Xla operation
  valueToOp[n.outputs().at(0)] = builder.ReduceWindowWithGeneralPadding(
      valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::MinValue(dataType)),
      OperatorRegistry::max(dataType), helper.getWindowDimensions(),
      helper.getWindowStrides(), helper.getInputPadding());
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(MaxPool, translateMaxPool)
}

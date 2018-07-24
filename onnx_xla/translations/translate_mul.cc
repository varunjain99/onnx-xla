#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translates Mul with broadcasting
// Implicit broadcasting if one operand has rank 0 or both have same rank
// Otherwise we explicity construct broadcastDims
onnxStatus translateMul(const Node& n,
                        XlaBuilder& builder,
                        ValueOpMap& valueToOp) {
  auto firstOp = valueToOp.at(n.inputs().at(0));
  auto secondOp = valueToOp.at(n.inputs().at(1));
  valueToOp[n.outputs().at(0)] = builder.Mul(
      firstOp, secondOp, OperatorRegistry::getMultidirectionalBroadcastArg(
                             builder, firstOp, secondOp));
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Mul, translateMul)
}

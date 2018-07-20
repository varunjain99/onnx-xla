#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Compute relu by taking max between the input and constant zero literal
onnxStatus translateRelu(const Node& n,
                         XlaBuilder& builder,
                         ValueOpMap& valueToOp) {
  auto input = valueToOp[n.inputs().at(0)];
  auto shape = builder.GetShape(input);
  if (!shape.ok()) {
    throw std::runtime_error("Internal error: Unexpected operation shape");
  }
  auto zero = builder.ConstantLiteral(
      *LiteralBase::CreateFromShape(shape.ValueOrDie()));
  auto maximum = builder.Max(input, zero);
  valueToOp[n.outputs().at(0)] = maximum;
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Relu, translateRelu)
}

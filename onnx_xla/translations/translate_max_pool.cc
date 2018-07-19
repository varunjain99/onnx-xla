#include "onnx_xla/operator_registry.h"

namespace onnx_xla {

onnxStatus translateMaxPool(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp) {
  // Set initValue and max
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());
  auto initValue = builder.ConstantLiteral(Literal::MinValue(dataType));
  XlaComputation max;
  {
    XlaBuilder builder("max");
    auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
    auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
    builder.Max(y, x);
    max = builder.Build().ConsumeValueOrDie();
  }
  auto inputOp = valueToOp.at(n.inputs().at(0));
  PoolHelper helper;
  auto status = helper.buildPoolOp(max, initValue, inputOp, builder, n);
  valueToOp[n.outputs().at(0)] = helper.poolOp;
  return status;
}
REGISTER_OPERATOR_TRANSLATOR(MaxPool, translateMaxPool)
}

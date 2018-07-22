/*#include "onnx_xla/operator_registry.h"

namespace onnx_xla {

template <typename nativeType>
XlaOp queueDivide(const Node& n,
                  XlaBuilder& builder,
                  const PoolHelper& helper) {
  auto kcount_include_pad = Symbol("count_include_pad");
  if (!n.hasAttribute(kcount_include_pad) || n.i(kcount_include_pad) != 0) {
    // pad included in averaged
    auto windowSize = std::accumulate(helper.windowDimensions.begin(),
                                      helper.windowDimensions.end(), 1L,
                                      std::multiplies<int64>());
    XlaOp divisor = builder.ConstantLiteral(
        *Literal::CreateR0<nativeType>(static_cast<nativeType>(windowSize)));
    std::cout << "WINDOW: " << windowSize << std::endl;
    return builder.Div(helper.poolOp, divisor);
  }
  // TODO: pad not included in average
  return XlaOp();
}

onnxStatus translateAveragePool(const Node& n,
                                XlaBuilder& builder,
                                ValueOpMap& valueToOp) {
  // Set initValue and max
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());
  auto initValue = builder.ConstantLiteral(Literal::Zero(dataType));
  XlaComputation add;
  {
    XlaBuilder builder("add");
    auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
    auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
    builder.Add(y, x);
    add = builder.Build().ConsumeValueOrDie();
  }
  auto inputOp = valueToOp.at(n.inputs().at(0));

  // Build add
  PoolHelper helper;
  auto poolStatus = helper.buildPoolOp(add, initValue, inputOp, builder, n);
  XlaOp averageOp;
  switch (dataType) {
    case xla::F32:
      averageOp = queueDivide<float>(n, builder, helper);
      break;
    case xla::F64:
      averageOp = queueDivide<double>(n, builder, helper);
      break;
    case xla::F16:
      averageOp = queueDivide<half>(n, builder, helper);
      break;
    case xla::BF16:
      averageOp = queueDivide<bfloat16>(n, builder, helper);
      break;
    case xla::C64:
      averageOp = queueDivide<complex64>(n, builder, helper);
      break;
    default:
      throw std::runtime_error("Pooling only defined for floating point types");
  }
  valueToOp[n.outputs().at(0)] = averageOp;
  return poolStatus;
}
REGISTER_OPERATOR_TRANSLATOR(AveragePool, translateAveragePool)
}*/

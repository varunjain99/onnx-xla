#include "onnx_xla/operator_registry.h"
#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {

onnxStatus translateAveragePool(const Node& n,
                                XlaBuilder& builder,
                                ValueOpMap& valueToOp) {
  // Set add TODO: softmax PR moves this to operator_registry.h
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());
  XlaComputation add;
  {
    XlaBuilder builder("add");
    auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
    auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
    builder.Add(y, x);
    add = builder.Build().ConsumeValueOrDie();
  }

  // Create ConvPoolHelper object (constructs attributes formatted for
  // XlaBuilder)
  ConvPoolHelper helper(n);

  // Enque a sum Xla operation
  XlaOp sumOp = builder.ReduceWindowWithGeneralPadding(
      valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::Zero(dataType)), add,
      helper.getWindowDimensions(), helper.getWindowStrides(),
      helper.getInputPadding());

  // Build average with implicit broadcasting (if count_include_pad != 0)
  // TODO: Support for count_include_pad (No test cases right now)
  auto kcount_include_pad = Symbol("count_include_pad");
  if (n.hasAttribute(kcount_include_pad) && n.i(kcount_include_pad) == 0) {
    throw std::runtime_error("count_include_pad = 0 not yet supported");
  }

  auto windowSize = std::accumulate(helper.getWindowDimensions().cbegin(),
                                    helper.getWindowDimensions().cend(), 1L,
                                    std::multiplies<int64>());
  auto divisorOp = ::tensorflow::FloatLiteral(&builder, dataType, windowSize);
  valueToOp[n.outputs().at(0)] = builder.Div(sumOp, divisorOp);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(AveragePool, translateAveragePool)
}

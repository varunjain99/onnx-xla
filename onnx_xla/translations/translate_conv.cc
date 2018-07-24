#include "onnx_xla/operator_registry.h"
#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {

onnxStatus translateConv(const Node& n,
                         XlaBuilder& builder,
                         ValueOpMap& valueToOp) {
  // Create ConvPoolHelper object (constructs attributes formatted for
  // XlaBuilder)
  ConvPoolHelper helper(n);

  // Enque corresponding Xla operation
  XlaOp convOp = builder.ConvGeneralDilated(
      valueToOp.at(n.inputs().at(0)), valueToOp.at(n.inputs().at(1)),
      helper.getWindowStrides(), helper.getInputPadding(), {},
      helper.getWindowDilations(),
      XlaBuilder::CreateDefaultConvDimensionNumbers(
          helper.getWindowStrides().size()));

  if (n.inputs().size() == 3 && n.inputs().at(2)->uniqueName() != "") {
    XlaOp biasOp = valueToOp.at(n.inputs().at(2));
    convOp = builder.Add(convOp, biasOp, {0});
  }
  valueToOp[n.outputs().at(0)] = convOp;

  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Conv, translateConv)
}

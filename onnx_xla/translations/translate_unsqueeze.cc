#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translate Unsqueeze by using XLA's Reshape
onnxStatus translateUnsqueeze(const Node& n,
                              XlaBuilder& builder,
                              ValueOpMap& valueToOp) {
  // Set origShape and axes
  std::vector<int64> origShape = OperatorRegistry::parseOnnxInputSizes(n, 0);

  if (!n.hasAttribute(kaxes)) {  // TODO ENFORCE
    std::cerr << "Missing Required Attribute" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  const auto& axes = n.is(kaxes);

  // Loop through both origShape and axes to make newShape
  auto origIndex = 0;
  auto axesIndex = 0;
  std::vector<int64> newShape;
  while (origIndex < origShape.size() && axesIndex < axes.size()) {
    if (axes[axesIndex] > origIndex + axesIndex) {
      newShape.emplace_back(origShape[origIndex++]);
    } else {
      newShape.emplace_back(1);
      axesIndex++;
    }
  }
  // Insert whatever remains to the end
  if (origIndex < origShape.size()) {
    newShape.insert(newShape.end(), origShape.begin() + origIndex,
                    origShape.end());
  } else if (axesIndex < axes.size()) {
    newShape.insert(newShape.end(), axes.size() - axesIndex, 1);
  }
  // Enqueue operation
  valueToOp[n.outputs().at(0)] =
      builder.Reshape(valueToOp.at(n.inputs().at(0)), newShape);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Unsqueeze, translateUnsqueeze)
}

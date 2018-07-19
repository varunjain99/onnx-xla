#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Compute Softmax
// 1) Find max of a batch
// 2) Subtract from current numbers
// 3) Exponentiate
// 3) Divide with implicit broadcasting
// TODO: Use and ENFORCE macro for checks
onnxStatus translateSoftmax(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp) {
  const Value* valueInput = n.inputs().at(0);
  auto inputOp = valueToOp.at(valueInput);
  auto dataType = onnxToPrimitive(valueInput->elemType());

  // Set axis value, defaulting to 1
  int64_t axis = 1;
  if (n.hasAttribute(kaxis)) {
    axis = n.i(kaxis);
  }
  if (axis < 0 || axis > valueInput->sizes().size()) {  // TODO: ENFORCE
    std::cerr << "Invalid axis attribute" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }

  // Set windowDimensions, which corresponds to a single batch
  if (valueInput->sizes().size() == 0) {  // TODO: ENFORCE
    std::cerr << "Invalid shape" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  std::vector<int64> windowDimensions;
  for (const auto& dimension : valueInput->sizes()) {
    if (!dimension.is_int) {  // TODO: ENFORCE
      std::cerr << "Invalid input shape dimension" << std::endl;
      return ONNXIFI_STATUS_INVALID_MODEL;
    }
    windowDimensions.emplace_back(dimension.dim);
  }
  for (auto i = 0; i < axis; ++i) {
    windowDimensions[i] = 1;
  }

  // windowStrides is all 1's
  std::vector<int64> windowStrides(valueInput->sizes().size(), 1);

  // Compute max of each batch
  auto maxOp = builder.ReduceWindow(
      inputOp, builder.ConstantLiteral(Literal::MinValue(dataType)),
      OperatorRegistry::max(dataType), windowDimensions, windowStrides,
      Padding::kValid);

  // Subtract max from each number (implict broadcasting)
  auto subOp = builder.Sub(inputOp, maxOp);

  // Exponentiate the result
  auto expOp = builder.Exp(subOp);

  // Sum up expOp for each batch
  auto dividendsOp = builder.ReduceWindow(
      expOp, builder.ConstantLiteral(Literal::Zero(dataType)),
      OperatorRegistry::add(dataType), windowDimensions, windowStrides,
      Padding::kValid);

  // Build softmax by dividing
  valueToOp[n.outputs().at(0)] = builder.Div(expOp, dividendsOp);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Softmax, translateSoftmax)
}

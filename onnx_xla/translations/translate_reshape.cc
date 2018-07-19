#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// TODO: ENFORCE_EQ macro
// Takes in data tensor and shape input
// Error if shape input is dynamic
// Parses ONNX shape input (can have 0's and one -1)
// Produces new_sizes attribute to make XlaOp for Reshape
onnxStatus translateReshape(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp,
                            const ValueLiteralMap& valueToLiteral) {
  // Check if shape input is static
  auto shapeDataIt = valueToLiteral.find(n.inputs().at(1));
  if (shapeDataIt == valueToLiteral.end()) {  // TODO: ENFORCE_EQ
    std::cerr << "Reshape operatore having dynamic shape input is not supported"
              << std::endl;
    return ONNXIFI_STATUS_UNSUPPORTED_VERSION;
  }
  // Compute # of elements in data tensor (check validity of shapes)
  const std::vector<Dimension>& originalShape = n.inputs().at(0)->sizes();
  int64_t numElements = 1;
  for (std::vector<Dimension>::const_iterator it = originalShape.begin();
       it != originalShape.end(); ++it) {
    if (!it->is_int) {
      std::cerr << "Reshape operator input data shape is not known"
                << std::endl;
      return ONNXIFI_STATUS_UNSUPPORTED_SHAPE;
    }
    numElements *= it->dim;
  }

  // Use static ONNX target shape data (from the constant literal constructed)
  // to build XLA target shape data and compute xlaProduct
  const tensorflow::gtl::ArraySlice<int64> onnxOperatorTargetShape =
      shapeDataIt->second->data<int64>();
  std::vector<int64> xlaOperatorTargetShape;
  int64 xlaProduct = 1;
  int64_t negativeOneIndex = -1;
  for (auto i = 0; i < onnxOperatorTargetShape.size(); ++i) {
    const int64& dim = onnxOperatorTargetShape.at(i);
    if (dim == -1) {
      if (negativeOneIndex > -1) {  // TODO: ENFORCE_GT
        std::cerr << "Invalid input: only one -1 allowed in input shape"
                  << std::endl;
        return ONNXIFI_STATUS_INVALID_MODEL;
      }
      negativeOneIndex = i;
      xlaOperatorTargetShape.emplace_back(-1);
    } else if (dim == 0) {
      xlaOperatorTargetShape.emplace_back(originalShape[i].dim);
    } else if (dim > 1) {
      xlaOperatorTargetShape.emplace_back(dim);
    } else {  // TODO: Enforce
      std::cerr << "Dimension value cannot be less than -1" << std::endl;
      return ONNXIFI_STATUS_INVALID_MODEL;
    }
    xlaProduct *= xlaOperatorTargetShape.back();
  }

  // Use numElements and xlaProduct to complete xlaOperatorTargetShape
  if (negativeOneIndex == -1) {
    if (xlaProduct != numElements) {  // TODO: enforce
      std::cerr << "Invalid shape to reshape to" << std::endl;
      return ONNXIFI_STATUS_INVALID_MODEL;
    }
  } else {
    if (numElements % (-xlaProduct) != 0) {  // TODO: enforce
      std::cerr << "Invalid shape to reshape to" << std::endl;
      return ONNXIFI_STATUS_INVALID_MODEL;
    }
    xlaOperatorTargetShape[negativeOneIndex] *= numElements / xlaProduct;
  }

  // Add resulting XlaOp to builder
  auto reshapeOp =
      builder.Reshape(valueToOp.at(n.inputs().at(0)), xlaOperatorTargetShape);
  valueToOp[n.outputs().at(0)] = reshapeOp;
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Reshape, translateReshape)
}

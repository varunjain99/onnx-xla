#include "onnx_xla/operator_registry.h"

namespace onnx_xla {

OperatorRegistry::OperatorRegisterOnce::OperatorRegisterOnce(
    const Symbol& nodeKind,
    TranslationFunction translator) {
  auto& map = OperatorRegistry::map();
  if (!map.insert(std::pair<Symbol, TranslationFunction>(nodeKind, translator))
           .second) {
    throw std::runtime_error("Registry error: Operator added more than once");
  }
}

onnxStatus OperatorRegistry::translate(const Node& n,
                                       XlaBuilder& builder,
                                       ValueOpMap& valueToOp) {
  auto& map = OperatorRegistry::map();
  auto it = map.find(n.kind());
  if (it != map.end()) {
    return it->second(n, builder, valueToOp);
  } else {
    std::cerr << "Operator translator not found" << std::endl;
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }
}

OperatorRegistry& OperatorRegistry::registry() {
  static OperatorRegistry registry_;
  return registry_;
}

TranslationMap& OperatorRegistry::map() {
  static TranslationMap map;
  return map;
}

onnxStatus PoolHelper::buildPoolOp(const XlaComputation& poolComp,
                                   const XlaOp& initValue,
                                   XlaOp& inputOp,
                                   XlaBuilder& builder,
                                   const Node& n) {
  // Set windowDimensions
  if (!n.hasAttribute(kkernel_shape)) {  // TODO: ENFORCE
    std::cerr << "Missing kernel_shape attribute" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  const auto& kernelShape = n.is(kkernel_shape);
  auto numAxes = kernelShape.size();
  windowDimensions.resize(numAxes + 2, 1);
  std::copy(kernelShape.begin(), kernelShape.end(),
            windowDimensions.begin() + 2);

  // Set windowStrides
  windowStrides.resize(numAxes + 2, 1);
  if (n.hasAttribute(kstrides)) {
    const auto& strides = n.is(kstrides);
    if (strides.size() != numAxes) {  // TODO: ENFORCE
      std::cerr << "Invalid dimension of strides attribute" << std::endl;
      return ONNXIFI_STATUS_INVALID_MODEL;
    }
    std::copy(strides.begin(), strides.end(), windowStrides.begin() + 2);
  }

  // Set padding
  padding.resize(2, std::pair<int64, int64>(0, 0));
  auto kauto_pad = Symbol("auto_pad");
  if (n.hasAttribute(kauto_pad)) {
    // Using auto_pad attribute
    auto autoPadType = n.s(kauto_pad);
    if (autoPadType == "VALID") {
      // No padding if VALID
      for (auto i = 0; i < numAxes; ++i) {
        padding.emplace_back(0, 0);
      }
    } else {
      // If not VALID, then SAME_UPPER or SAME_LOWER
      // Translate input sizes into int64_t vector
      std::vector<int64_t> inputSizes;
      const std::vector<Dimension>& sizes = n.inputs().at(0)->sizes();
      for (auto it = sizes.begin(); it != sizes.end(); ++it) {
        if (!it->is_int) {  // TODO: ENFORCE
          std::cerr << "Invalid dimension of input tensor" << std::endl;
          return ONNXIFI_STATUS_INVALID_MODEL;
        }
        inputSizes.emplace_back(it->dim);
      }
      // Compute axisPadding using formula in docs for SAME_UPPER and SAME_LOWER
      std::vector<int64_t> axisPadding;
      for (auto i = 0; i < numAxes; ++i) {
        auto outputShape =
            (inputSizes.at(i + 2) + windowStrides.at(i + 2) - 1) /
            windowStrides.at(i + 2);
        axisPadding.emplace_back((outputShape - 1) * windowStrides.at(i + 2) +
                                 windowDimensions.at(i + 2) -
                                 inputSizes.at(i + 2));
      }
      if (autoPadType == "SAME_UPPER") {
        // Emplace (floor, ceiling) of axisPadding/2
        for (auto i = 0; i < numAxes; ++i) {
          padding.emplace_back(axisPadding.at(i) / 2,
                               (axisPadding.at(i) + 1) / 2);
        }
      } else if (autoPadType == "SAME_LOWER") {
        // Emplace (ceiling, floor) of axisPadding/2
        for (auto i = 0; i < numAxes; ++i) {
          padding.emplace_back((axisPadding.at(i) + 1) / 2,
                               axisPadding.at(i) / 2);
        }
      } else {  // TODO: ENFORCE
        std::cerr << "Invalid auto_pad attribute" << std::endl;
        return ONNXIFI_STATUS_INVALID_MODEL;
      }
    }
  } else {
    // Not using auto_pads attribute
    if (!n.hasAttribute(kpads)) {
      // If not pads attribute, default to no padding
      for (auto i = 0; i < numAxes; ++i) {
        padding.emplace_back(0, 0);
      }
    } else {
      // If pads attribute, fill in padding with pairs from pads
      auto pads = n.is(kpads);
      for (auto i = 0; i < numAxes; ++i) {
        padding.emplace_back(pads.at(i), pads.at(i + numAxes));
      }
    }
  }
  // Set the XlaOp
  poolOp = builder.ReduceWindowWithGeneralPadding(
      inputOp, initValue, poolComp, windowDimensions, windowStrides, padding);
  return ONNXIFI_STATUS_SUCCESS;
}
}

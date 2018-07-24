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

std::vector<int64> OperatorRegistry::getMultidirectionalBroadcastArg(
    const XlaBuilder& builder,
    const XlaOp& firstOp,
    const XlaOp& secondOp) {
  auto firstNDim = ShapeUtil::Rank(builder.GetShape(firstOp).ValueOrDie());
  auto secondNDim = ShapeUtil::Rank(builder.GetShape(secondOp).ValueOrDie());
  std::vector<int64> broadcastDims;
  if (firstNDim != secondNDim || firstNDim != 0 || secondNDim != 0) {
    auto minDim = std::min(firstNDim, secondNDim);
    auto maxDim = std::max(firstNDim, secondNDim);
    for (auto j = 0; j < minDim; ++j) {
      broadcastDims.push_back(j + maxDim - minDim);
    }
  }
  return broadcastDims;
}
}

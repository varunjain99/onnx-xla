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

XlaComputation OperatorRegistry::add(PrimitiveType dataType) {
  XlaBuilder builder("add");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
  builder.Add(y, x);
  return builder.Build().ConsumeValueOrDie();
}

XlaComputation OperatorRegistry::max(PrimitiveType dataType) {
  XlaBuilder builder("max");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
  builder.Max(y, x);
  return builder.Build().ConsumeValueOrDie();
}

std::vector<int64> OperatorRegistry::parseOnnxInputSizes(const Node& n,
                                                         size_t inputIndex) {
  if (!n.inputs().at(inputIndex)->has_sizes()) {  // TODO: Enforce
    throw std::runtime_error("Missing shape");
  }
  std::vector<int64> shapeInts;
  const auto& shapeDims = n.inputs().at(inputIndex)->sizes();
  for (const auto& dimension : shapeDims) {
    if (!dimension.is_int) {  // TODO: Enforce
      throw std::runtime_error("Invalid dimension");
    }
    shapeInts.emplace_back(dimension.dim);
  }
  return shapeInts;
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

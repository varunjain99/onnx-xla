#include "onnx_xla/operator_registry.h"

namespace onnx_xla  {
  // TODO: test
  // Do successive additions to the sum operation,
  // Satisfy XLA broadcast semantics:
  // 1) If same rank or scalar, XLA does broadcasting w/o explicit broadcasting dimensions
  // 2) Otherwise, use numpy matching of trailing dimensions
  onnxStatus translateSum(const Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    auto firstOp = valueToOp.at(n.inputs().at(0));
    for (auto i = 1; i < n.inputs().size(); ++i)  {
      auto secondOp = valueToOp.at(n.inputs().at(i));
      auto firstNDim = ShapeUtil::Rank(builder.GetShape(firstOp).ValueOrDie());
      auto secondNDim = ShapeUtil::Rank(builder.GetShape(secondOp).ValueOrDie());
      std::vector<int64> broadcastDims;
      if (firstNDim != secondNDim || firstNDim != 0 || secondNDim != 0)  {
        auto minDim = std::min(firstNDim, secondNDim);
        auto maxDim = std::max(firstNDim, secondNDim);
        for (auto j = 0; j < minDim; ++j)  {
          broadcastDims.push_back(j + maxDim - minDim);
        }
      }
      firstOp = builder.Add(firstOp, secondOp, broadcastDims);
    }
    valueToOp[n.outputs().at(0)] = firstOp;
    return ONNXIFI_STATUS_SUCCESS;
  }
  REGISTER_OPERATOR_TRANSLATOR(Sum, translateSum)
}


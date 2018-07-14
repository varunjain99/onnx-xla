#include "onnx_xla/operator_registry.h"
#include "onnx_xla/types.h"
#include <algorithm>

namespace onnx_xla  {
  
  OperatorRegistry::OperatorRegisterOnce::OperatorRegisterOnce(const Symbol& nodeKind, TranslationFunction translator)  {
    auto& map = OperatorRegistry::map();
    if (!map.insert(std::pair<Symbol, TranslationFunction>(nodeKind, translator)).second)  {
      throw std::runtime_error("Registry error: Operator added more than once");
    }
  }

  onnxStatus OperatorRegistry::translate(const Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    auto& map = OperatorRegistry::map();
    auto it = map.find(n.kind());
    if (it != map.end()) {
      return it->second(n, builder, valueToOp);
    } else {
      return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
    }
  }

  OperatorRegistry& OperatorRegistry::registry()  {
    static OperatorRegistry registry_;
    return registry_;
  }

  TranslationMap& OperatorRegistry::map()  {
    static TranslationMap map;
    return map;
  }

  onnxStatus translateRelu(const Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    auto input = valueToOp[n.inputs().at(0)];
    auto shape = builder.GetShape(input);
    if (!shape.ok())  {
      throw std::runtime_error("Internal error: Unexpected operation shape");
    }
    auto zero = builder.ConstantLiteral(*LiteralBase::CreateFromShape(shape.ValueOrDie()));
    auto maximum = builder.Max(input, zero);
    valueToOp[n.outputs().at(0)] = maximum;
    return ONNXIFI_STATUS_SUCCESS;
  }
  REGISTER_OPERATOR_TRANSLATOR(Relu, translateRelu)

  //To be tested
  //If 1 input, match output value to same op input value is matched to
  //Otherwise do successive additions, being sure to satisfy broadcast semantics
    //If same rank or scalar, no need to specify
    //otherwise, use numpy matching of trailing dimensions
  onnxStatus translateSum(const Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    if (n.inputs().size() == 1)  {
      valueToOp[n.outputs().at(0)] = valueToOp[n.inputs().at(0)];
      return ONNXIFI_STATUS_SUCCESS;
    }
    auto firstOp = valueToOp[n.inputs().at(0)];
    for (auto i = 1; i < n.inputs().size(); ++i)  {
      auto secondOp = valueToOp[n.inputs().at(i)];
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
 
  //TODO: Handle Undefined properly
  onnxStatus translateUndefined(const Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    return ONNXIFI_STATUS_SUCCESS;
  }
  REGISTER_OPERATOR_TRANSLATOR(Undefined, translateUndefined) 
}

#include "onnx_xla/operator_registry.h"
#include <iostream>
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
    valueToOp[n.outputs().at(0)] = std::move(maximum);
    return ONNXIFI_STATUS_SUCCESS;
  }
  REGISTER_OPERATOR_TRANSLATOR(Relu, translateRelu)
 
  //TODO: Handle Undefined properly
  onnxStatus translateUndefined(const Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    return ONNXIFI_STATUS_SUCCESS;
  }
  REGISTER_OPERATOR_TRANSLATOR(Undefined, translateUndefined) 
}

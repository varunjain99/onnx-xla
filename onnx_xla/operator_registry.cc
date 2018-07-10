#include "onnx_xla/operator_registry.h"
namespace onnx_xla  {
  
  OperatorRegistry::OperatorRegisterOnce::OperatorRegisterOnce(const Symbol& nodeKind, TranslationFunction translator)  {
    auto& map = OperatorRegistry::map();
    if (map.find(nodeKind) == map.end())  {
      map[nodeKind] = translator;
    } else {
      throw("Operator registry error");
    }
  }

  onnxStatus OperatorRegistry::executeTranslation(Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    auto& map = OperatorRegistry::map();
    if (map.find(n.kind()) != map.end())  {
      return map[n.kind()](n, builder, valueToOp);
    } else {
      return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
    }
  }

  OperatorRegistry* OperatorRegistry::registry()  {
    static OperatorRegistry registry_;
    return &registry_;
  }

  TranslationMap& OperatorRegistry::map()  {
    static TranslationMap map;
    return map;
  }

  onnxStatus translateRelu(Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    auto input = valueToOp[n.inputs()[0]];
    auto shape = builder.GetShape(input);
    if (!shape.ok())  {
      throw("Invalid shape of input to relu");
    }
    auto zero = builder.ConstantLiteral(*LiteralBase::CreateFromShape(shape.ValueOrDie()));
    auto maximum = builder.Max(input, zero);
    valueToOp[n.outputs()[0]] = std::move(maximum);
    return ONNXIFI_STATUS_SUCCESS;
  }
  REGISTER_OPERATOR_TRANSLATOR(Relu, translateRelu)
 
  //TODO: Handle Undefined properly
  onnxStatus translateUndefined(Node& n, XlaBuilder& builder, ValueOpMap& valueToOp)  {
    return ONNXIFI_STATUS_SUCCESS;
  }
  REGISTER_OPERATOR_TRANSLATOR(Undefined, translateUndefined) 
}

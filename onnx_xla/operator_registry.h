#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/onnxifi.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"

#include <functional>

namespace onnx_xla  {

  using xla::Literal;
  using xla::ShapeUtil;
  using xla::Shape;
  using xla::primitive_util::NativeToPrimitiveType;
  using xla::XlaOp;
  using xla::XlaBuilder;
  using xla::LiteralBase;
  using xla::StatusOr;

  using ONNX_NAMESPACE::Tensor;
  using ONNX_NAMESPACE::Value;
  using ONNX_NAMESPACE::Dimension;
  using ONNX_NAMESPACE::Graph;
  using ONNX_NAMESPACE::Symbol;
  using ONNX_NAMESPACE::Node;

  using ValueOpMap = std::unordered_map<const Value*, XlaOp>;
  using TranslationFunction = std::function<onnxStatus(Node&, XlaBuilder&, ValueOpMap&)>;
  using TranslationMap = std::unordered_map<Symbol, TranslationFunction>;

  class OperatorRegistry final {
  public:
    class OperatorRegisterOnce final  {
    public:
      OperatorRegisterOnce(const Symbol& nodeKind, TranslationFunction translator);     
    };
 
    onnxStatus executeTranslation(Node& n, XlaBuilder& builder, ValueOpMap& valueToOp);
    static OperatorRegistry* registry();
  private:
    OperatorRegistry() = default;
    static TranslationMap& map();
  };

  #define REGISTER_OPERATOR_TRANSLATOR(name, translator)                                     \
    static OperatorRegistry::OperatorRegisterOnce register##name(Symbol(#name), translator); \

}

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
  
  //Class for registry of ONNX operators with corresponding translation functions
  class OperatorRegistry final {
  public:
    //Use constructor (through macro) to register translator at static time
    class OperatorRegisterOnce final  {
    public:
      OperatorRegisterOnce(const Symbol& nodeKind, TranslationFunction translator);     
    };
 
    //Execute translation for given node
    onnxStatus executeTranslation(Node& n, XlaBuilder& builder, ValueOpMap& valueToOp);
    //Returns pointer to static singleton registry
    static OperatorRegistry* registry();
  private:
    //Singleton instance should only be made in the class
    OperatorRegistry() = default;
    //Wrapper for registry map - should not be directly accessed
      //Register in map through below macro
      //Execute translation function through registry()->executeTranslation
    static TranslationMap& map();
  };
  
  //Use this macro to register Symbol("name") with translator of type TranslationFunction 
  #define REGISTER_OPERATOR_TRANSLATOR(name, translator)                                     \
    static OperatorRegistry::OperatorRegisterOnce register##name(Symbol(#name), translator); \

}

#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "onnx/onnx/common/ir.h"
#include "onnx-xla/onnx_xla/type_conversion.h"


using ONNX_NAMESPACE;
using xla;

namespace onnx-xla {
  class XlaTransform final  {
  public:
    XlaTransform(const Graph& ir_, const string name) :
      ir_(ir), builder_(XlaBuilder(name)), global_param_number(0) {}

    ~XlaTransform() {
      for (const auto& l : literals_) {
        delete l;
      }
    }
  private:
    const Graph& ir_;
    XlaBuilder builder_;
    vector<Literal*> literals_;
    std::unordered_map<const Value*, XlaOp*> value_to_op_;
    std::unordered_map<const Value*, Shape> value_to_shape_;
    std::unordered_map<const Value*, PrimitiveType> value_to_primitive_type_;
    int64 global_param_number;

    //returns pointer to Literal, NULL if error
    Literal* initializerToLiteral(const Tensor& t);
    //true if successful, false if not
    bool intializersToLiterals();
    void computeBuild();
    void registerOutputWithOp(const Value* v, XlaOp* op);
    void registerOutputWithOp(const Value* v, XlaOp* op);
  }
}

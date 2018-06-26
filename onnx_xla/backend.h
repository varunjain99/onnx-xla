#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "onnx/onnx/common/ir.h"
#include "onnx-xla/onnx_xla/type_conversion.h"


using ONNX_NAMESPACE;
using xla;

namespace onnx-xla {

  struct conversion_error final : public std::exception {
  private:
    const std::string msg_;
  public:
    explicit conversion_error(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }
  };

  class XlaTransform final  {
  public:
    XlaTransform(const Graph& ir_, const string name) :
      ir_(ir), builder_(XlaBuilder(name)), computation_(NULL),
      global_param_number(0) {}

    ~XlaTransform() {
      for (const auto& l : literals_) {
        delete l;
      }
    }
  private:
    const Graph& ir_;
    XlaBuilder builder_;
    XlaComputation* computation_;
    vector<std::unique_ptr<Literal>> literals_;
    std::unordered_map<const Value*, XlaOp*> value_to_op_;
    int64 global_param_number;

    Literal* initializerToLiteral(const Tensor& t);
    void intializersToLiterals();
    Shape shapeOfValue(const Value* v);
    void registerOutputWithOp(const Value* v, XlaOp* op);
    void registerOutputWithOp(const Value* v, XlaOp* op, int index);
    void buildComputation();

  }
}

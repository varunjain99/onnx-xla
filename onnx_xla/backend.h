#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "onnx/common/ir.h"
#include "onnx_xla/type_conversion.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <memory>

namespace onnx_xla {

  struct conversion_error final : public std::exception {
  private:
    const std::string msg_;
  public:
    explicit conversion_error(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }
  };

  class XlaTransform final  {
  public:
    XlaTransform(ONNX_NAMESPACE::Graph& ir, const std::string name) :
      ir_(ir), builder_(name), global_param_number(0) {}

    ~XlaTransform() {}
  private:
    ONNX_NAMESPACE::Graph& ir_;
    xla::XlaBuilder builder_;
    std::vector<std::unique_ptr<xla::Literal>> literals_;
    std::unordered_map<const ONNX_NAMESPACE::Value*, xla::XlaOp> value_to_op_;
    xla::int64 global_param_number;

    std::unique_ptr<xla::Literal> initializerToLiteral(const ONNX_NAMESPACE::Tensor& t);
    void intializersToLiterals();
    xla::Shape shapeOfValue(const ONNX_NAMESPACE::Value* v);
    void registerValueOp(const ONNX_NAMESPACE::Value* v, xla::XlaOp& op);
    void registerValueOp(const ONNX_NAMESPACE::Value* v, xla::XlaOp& op, int index);
    void translateGraph();
    std::vector<xla::Literal> executeComputation();
  };
}

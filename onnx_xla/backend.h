#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "onnx_xla/types.h"
#include "onnx/common/ir.h" 

#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include <grpcpp/grpcpp.h>

#include <memory>

namespace onnx_xla {
   using xla::Literal;
   using xla::ShapeUtil;
   using xla::Shape;
   using xla::primitive_util::NativeToPrimitiveType;
   using xla::XlaOp;
   using xla::GlobalData;
   using xla::LiteralBase;
   using xla::StatusOr;

   using ONNX_NAMESPACE::Tensor;
   using ONNX_NAMESPACE::Value;
   using ONNX_NAMESPACE::Dimension;
   using ONNX_NAMESPACE::Graph;
   using ONNX_NAMESPACE::Symbol;

  struct conversion_error final : public std::exception {
  private:
    const std::string msg_;
  public:
    explicit conversion_error(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }
  };

  class XlaTransform final  {
  public:
    XlaTransform(ONNX_NAMESPACE::Graph& ir,
                 const std::string& build_name); 
    ~XlaTransform() {}
    void initializersToLiterals();
    void sendLiterals();
    void translateGraph();
    std::vector<xla::Literal> executeComputation();
  
  private:
    ONNX_NAMESPACE::Graph& ir_;
    xla::XlaBuilder builder_;
    std::vector<std::unique_ptr<xla::Literal>> literals_;
    std::vector<GlobalData*> arguments_;
    std::unordered_map<const ONNX_NAMESPACE::Value*, xla::XlaOp> value_to_op_;
    xla::int64 global_param_number_;

    std::unique_ptr<xla::Literal> initializerToLiteral(const ONNX_NAMESPACE::Tensor& t);
    xla::Shape shapeOfValue(const ONNX_NAMESPACE::Value* v);
    void registerValueOp(const ONNX_NAMESPACE::Value* v, xla::XlaOp& op);
    void registerValueOp(const ONNX_NAMESPACE::Value* v, xla::XlaOp& op, int index);
  };
}

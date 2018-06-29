#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "onnx_xla/types.h"
#include "onnx/common/ir.h"
#include "onnx/onnxifi.h"

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

  class XlaTransform;
  class XlaExecution;

  class XlaExecutor final  {
  public:
    void sendLiterals();
    std::vector<xla::Literal> executeComputation();
  private:
    xla::XlaComputation computation_;
    std::vector<std::unique_ptr<xla::Literal>> static_literals_;
    std::vector<GlobalData*> arguments_;
    uint32_t num_inputs_;
    uint32_t num_outputs_;
    std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto_DataType> io_data_type_;
    std::unordered_map<std::string, vector<Dimension>> io_shape_;
    std::unordered_map<std::string, onnxPointer> input_buffers_;
    std::unordered_map<std::string, onnxPointer> output_buffers_;
    std::vector<std::string> output_names_;

    std::unique_ptr<xla::Literal> tensorToLiteral(const ONNX_NAMESPACE::Tensor& t);
    std::unique_ptr<xla::Literal> inputToLiteral(const std::string& name);
    void initIO(uint32_t inputsCount, onnxTensorDescriptor* inputDescriptors,
                uint32_t  outputsCount, onnxTensorDescriptor* outputDescriptors);
    friend class XlaTransform;
  };

  class XlaTransform final  {
  public:
    XlaTransform(ONNX_NAMESPACE::Graph& ir,
                 const std::string& build_name);
    ~XlaTransform();
    void translateGraph();
    XlaExecutor* executor();

  private:
    ONNX_NAMESPACE::Graph& ir_;
    xla::XlaBuilder builder_;
    XlaExecutor* executor_;
    std::unordered_map<const ONNX_NAMESPACE::Value*, xla::XlaOp> value_to_op_;
    xla::int64 global_param_number_;

    xla::Shape shapeOfValue(const ONNX_NAMESPACE::Value* v);
    void registerValueOp(const ONNX_NAMESPACE::Value* v, xla::XlaOp& op);
    void registerValueOp(const ONNX_NAMESPACE::Value* v, xla::XlaOp& op, int index);
    void fillIOMetadata();
  };

}

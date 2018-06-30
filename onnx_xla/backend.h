#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "onnx_xla/types.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/onnxifi.h"
#include "onnx/proto_utils.h"
#include "onnx/shape_inference/implementation.h"

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
  using xla::XlaBuilder;
  using xla::GlobalData;
  using xla::LiteralBase;
  using xla::StatusOr;
  using xla::XlaComputation;

  using ONNX_NAMESPACE::Tensor;
  using ONNX_NAMESPACE::Value;
  using ONNX_NAMESPACE::Dimension;
  using ONNX_NAMESPACE::Graph;
  using ONNX_NAMESPACE::Symbol;
  using ONNX_NAMESPACE::ModelProto;

  struct conversion_error final : public std::exception {
  private:
    const std::string msg_;
  public:
    explicit conversion_error(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }
  };

  class XlaTransform;
  class XlaExecutor;

  class XlaExecutor final  {
  public:
    void initIO(uint32_t inputsCount, const onnxTensorDescriptor* inputDescriptors,
                uint32_t  outputsCount, const onnxTensorDescriptor* outputDescriptors);
    void sendLiterals();
    void executeComputation();
  private:
    XlaComputation computation_;
    std::vector<std::unique_ptr<Literal>> static_literals_;
    std::vector<GlobalData*> arguments_;
    uint32_t num_inputs_;
    uint32_t num_outputs_;
    std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto_DataType> io_data_type_;
    std::unordered_map<std::string, std::vector<Dimension>> io_shape_;
    std::unordered_map<std::string, onnxPointer> input_buffers_;
    std::unordered_map<std::string, onnxPointer> output_buffers_;
    std::vector<std::string> output_names_;

    std::unique_ptr<Literal> tensorToLiteral(const Tensor& t);
    std::unique_ptr<Literal> inputToLiteral(const std::string& name);
    
    friend class XlaTransform;
  };

  class XlaTransform final  {
  public:
    XlaTransform(std::unique_ptr<Graph> ir,
                 const std::string& build_name);
    ~XlaTransform();
    void translateGraph();
    XlaExecutor* executor();

  private:
    std::unique_ptr<Graph> ir_;
    XlaBuilder builder_;
    std::unique_ptr<XlaExecutor> executor_;
    std::unordered_map<const Value*, XlaOp> value_to_op_;
    int64 global_param_number_;

    Shape shapeOfValue(const Value* v);
    void registerValueOp(const Value* v, XlaOp& op);
    void registerValueOp(const Value* v, XlaOp& op, int index);
    void fillIOMetadata();
  };

  class OnnxParser {
  public:
    OnnxParser(const void* serializedModel, size_t serializedModelSize);
    std::unique_ptr<Graph> parse();
  private:
    ModelProto model_;
  };
}

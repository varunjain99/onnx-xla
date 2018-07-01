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
  class OnnxParser;

  //Engine to execute an XlaComputation constructed by XlaTransform. The computation_
  //and static_literals_ are filled by the XlaTransform object. To run, call initIO to
  //verify IO metadata and to send IO locations. Once IO data is present, execute sendLiterals
  //and executeComputation to run. If successful, output tensors will be present at the 
  //output_buffers_ pointers.
  class XlaExecutor final  {
  public:
    //initIO: used to pass I/O tensor values and metadata to the engine
    void initIO(uint32_t inputsCount, const onnxTensorDescriptor* inputDescriptors,
                uint32_t  outputsCount, const onnxTensorDescriptor* outputDescriptors);

    //sendLiterals: sends constants and input tensors to the server
    void sendLiterals();

    //executeComputation: runs the computation on the server using passed data 
    void executeComputation();

  private:
    //computation_: computation to be run
    XlaComputation computation_;

    //static_literals_: weights passed from initializers
    std::vector<std::unique_ptr<Literal>> static_literals_;

    //arguments_: weight and IO data to be passed to the server
    std::vector<GlobalData*> arguments_;

    //Store IO metadata to 
    //  Verify IO has correct shape, data type, TODO: memory type
    //  Get input and output locations
    uint32_t num_inputs_;
    uint32_t num_outputs_;
    std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto_DataType> io_data_type_;
    std::unordered_map<std::string, std::vector<Dimension>> io_shape_;
    std::unordered_map<std::string, onnxPointer> input_buffers_;
    std::unordered_map<std::string, onnxPointer> output_buffers_;

    //output_names_: order of this vector corresponds to order of returned literals
    std::vector<std::string> output_names_;

    //Helper functions to translate weights and inputs to literals
    std::unique_ptr<Literal> tensorToLiteral(const Tensor& t);
    std::unique_ptr<Literal> inputToLiteral(const std::string& name);
    
    friend class XlaTransform;
  };


  //Engine to transform an IR graph to a form that can be executed by the XLA server.
  //When the object is constructed, ownership of the IR graph is passed to it. Then,
  //execute translateGraph to build up the XlaExecutor object (and thus the XlaComputation.
  //To get the handle on the XlaExecutor that is constructed, call executor() (can only be
  //done once).
  class XlaTransform final  {
  public:
    //Passes IR graph to be transformed and name of builder
    //TODO: Remove build_name? or keep for debugging purposes?
    XlaTransform(std::unique_ptr<Graph> ir,
                 const std::string& build_name);
    ~XlaTransform();

    //Fills up XlaExecutor based on the IR graph given
    //  Fills up executor_->static_literals_ from initializers
    //  Fills up executor_'s expected IO metadata, which can be verified in initIO
    //  Translates IR graph node by node 
    //    TODO: Fix kUndefinded translation, which is present for relu test
    //  Fills up exector_'s output names
    void translateGraph();

    //Used to get handle to XlaExecutor
    //NOTE: Can only be called on once as it releases the unique pointer. Freeing
    //memory must be handled by caller.
    XlaExecutor* executor();

  private:
    //IR graph to be translated
    std::unique_ptr<Graph> ir_;

    //Builder that builds XlaComputation
    //  TODO: Remove? Currently only used by one function
    XlaBuilder builder_;

    //XlaExecutor that is built up to do the computation later
    std::unique_ptr<XlaExecutor> executor_;

    //Used to keep track of values and XlaOp's
    //whose output corresponds to them
    std::unordered_map<const Value*, XlaOp> value_to_op_;

    //Keeps track of number of parameters in computation
    int64 global_param_number_;

    //Helper to get shape of associated value
    //TODO: make static
    Shape shapeOfValue(const Value* v);

    //Helpers to register value with and operation in value_to_op_
    void registerValueOp(const Value* v, XlaOp& op);
    void registerValueOp(const Value* v, XlaOp& op, int index);

    //Fill executor_'s IO metadata (type, shape) to be verified later by
    //the executor_
    void fillIOMetadata();
  };

  //Engine to build up an IR graph from proto bytes format. Model validation and 
  //shape inference for entire graph is conducted here.
  class OnnxParser {
  public:
    //Constructs ModelProto from byte form
    OnnxParser(const void* serializedModel, size_t serializedModelSize);

    //Shape inference and model validation, followed by conversion to IR
    std::unique_ptr<Graph> parse();
  private:
    ModelProto model_;
  };
}

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
  //is filled by the XlaTransform object. To run, call initIO to verify IO
  //metadata and to declare IO locations. Once IO data is present, execute sendInputs
  //and executeComputation to run. If successful, output tensors will be present at the 
  //output_buffers_ pointers.
  class XlaExecutor final  {
  public:
    //Used to pass IO metadata and locations to the engine
    void initIO(uint32_t inputsCount, const onnxTensorDescriptor* inputDescriptors,
                uint32_t  outputsCount, const onnxTensorDescriptor* outputDescriptors);

    //Sends input tensor values to the server
    void sendInputs();

    //Runs the computation on the server using passed input 
    void executeComputation();

  private:
    //computation to be run
    XlaComputation computation_;

    //Input data to be passed to the server
      //Filled in by sendInputs
      //Used by executeComputation
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

    //Mapping of parameter number to input name; use to fill arguments_ in the correct order
    std::vector<std::string> param_input_name_;

    //Used to copy output returned from XLA to output buffers 
    std::vector<std::string> output_names_;

    //Helper functions to translate tensors, inputs, and weights to literals
    std::unique_ptr<Literal> tensorToLiteral(const Tensor& t);
    std::unique_ptr<Literal> inputNameToLiteral(const std::string& name);
    std::unique_ptr<Literal> descriptorToLiteral(const onnxTensorDescriptor& t);
    
    friend class XlaTransform;
  };


  //Engine to transform an IR graph to a form that can be executed by the XLA server.
  //When the object is constructed, ownership of the IR graph is passed to it. Then,
  //execute translateGraph to build up the XlaExecutor object (and thus the XlaComputation).
  //To get the handle on the XlaExecutor that is constructed, call executor() (can only be
  //done once).
  class XlaTransform final  {
  public:
    //Passes IR graph to be transformed, name of builder, and weightDescriptor info
    //TODO: Remove build_name? or keep for debugging purposes?
    XlaTransform(std::unique_ptr<Graph> ir,
                 const std::string& build_name,
                 uint32_t weightsCount, 
                 const onnxTensorDescriptor *weightDescriptors);
    ~XlaTransform();

    //Fills up XlaExecutor based on the IR graph. Function accomplishes:
    //  Initializer/weight values added as constants to the graph 
    //  Fills up executor_'s expected IO metadata, which can be verified in initIO
    //  Translates IR graph node by node 
    //    TODO: Fix kUndefinded translation, which is present for relu test
    //    TODO: Make function registry
    //  Fills up exector_'s output names
    void translateGraph();

    //Used to get handle to XlaExecutor
    //NOTE: Can only be called on once as it releases the unique pointer. Freeing
    //memory must be handled by caller.
    XlaExecutor* executor();

  private:
    //IR graph to be translated
    std::unique_ptr<Graph> ir_;

    //Weight Descriptor information
    uint32_t weights_count_;
    const onnxTensorDescriptor* weight_descriptors_;

    //Builder that builds XlaComputation
    //  TODO: Remove? Currently only used by one function
    XlaBuilder builder_;

    //XlaExecutor that is built up to do the computation later
    std::unique_ptr<XlaExecutor> executor_;

    //Used to keep track of values and XlaOp's
    //whose output corresponds to them
    std::unordered_map<const Value*, XlaOp> value_to_op_;

    //Keeps track of number of parameters in computation
    //  TODO: Make local? Only used by one function
    int64 global_param_number_;

    //Helper to get shape of associated value
    static inline Shape shapeOfValue(const Value* v);

    //Helpers to register value with and operation in value_to_op_
    void registerValueOp(const Value* v, XlaOp& op);
    void registerValueOp(const Value* v, XlaOp& op, int index);


    //Create ConstantLiteral XlaOps for initializers/weights, verifying weight descriptors;
    //Creates params for other runtime inputs
    //Fill executor_'s input metadata (type, shape) to be verified later
    void handleInputs();

    //Fill output_names_
    //Create output XlaOp
    //Fill executor_'s output metadata (type, shape) to be verified later
    void handleOutputs();
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

#include "onnx_xla/backend.h"

namespace onnx_xla {
  #define SWITCH(data_type)                                                    \
    switch(data_type) {                                                        \
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {                      \
        OPERATION(float, float, floats)                                        \
      }                 						       \
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {		       \
        OPERATION(complex64, complex64, floats)    			       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:  {		       \
        OPERATION(int32_t, half, int32s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:  {		       \
        OPERATION(int32_t, bool, int32s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:  {		       \
        OPERATION(int32_t, int8, int32s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:  {		       \
        OPERATION(int32_t, int16, int32s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:  {		       \
        OPERATION(int32_t, int32, int32s)				       \
      }         							       \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:  {		       \
        OPERATION(int32_t, uint8, int32s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {		       \
        OPERATION(int32_t, uint16, int32s)			               \
      }          							       \
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {		       \
        OPERATION(int64_t, int64, int64s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:  {		       \
        OPERATION(uint64_t, uint32, uint64s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {		       \
        OPERATION(uint64_t, uint64, uint64s)				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:  {                     \
        OPERATION(double, double, doubles) 				       \
      } 								       \
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:		       \
      case ONNX_NAMESPACE::TensorProto_DataType_STRING:			       \
      case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:		       \
      default:  {							       \
        throw conversion_error("Tensor not of a convertible data type.");      \
      }  								       \
    }  									       \

  std::unique_ptr<Literal> XlaExecutor::tensorToLiteral(const Tensor& t) {

    #define OPERATION(type_from, type_to, vec)                                 \
      type_from* t_data;                                                       \
      if (t.is_raw_data())  {                                                  \
        t_data = (type_from*) t.raw().c_str();                                 \
      } else {                                                                 \
        t_data = (type_from*) t.vec().data();                                  \
      }                                                                        \
      std::vector<int64> sizes;                                                \
      for (auto n : t.sizes()) {                                               \
        sizes.push_back(n);                                                    \
      }                                                                        \
      auto l = std::unique_ptr<Literal>(new Literal(ShapeUtil::MakeShape(      \
               NativeToPrimitiveType<type_to>(), sizes)));                     \
      int64 num_elements = std::accumulate(sizes.begin(),                      \
                                             sizes.end(),                      \
			    (int64) 1, std::multiplies<int64>());              \
      tensorflow::gtl::MutableArraySlice<type_to> l_data = l->data<type_to>(); \
      for (auto i = 0; i < num_elements; ++i) {                                \
        l_data[i] = (type_to) t_data[i];                                       \
      }									       \
      return l;                                                                \
    
    SWITCH(t.elem_type())
    #undef OPERATION
  }

  std::unique_ptr<Literal> XlaExecutor::inputToLiteral(const std::string& name) {
    std::vector<int64> sizes;
    for (auto i = 0; i < io_shape_[name].size(); ++i) {  
      sizes.push_back((int64) io_shape_[name][i].dim);    
    }
    int64 num_elements = std::accumulate(sizes.begin(), sizes.end(),    
                                         (int64) 1, std::multiplies<int64>());           

    #define OPERATION(type_from, type_to, vec)                                 \
      auto l = std::unique_ptr<Literal>(new Literal(ShapeUtil::MakeShape(      \
               NativeToPrimitiveType<type_to>(), sizes)));                     \
      tensorflow::gtl::MutableArraySlice<type_to> l_data = l->data<type_to>(); \
      ONNX_ASSERT(input_buffers_[name]);                                       \
      type_from* inputData = (type_from*) input_buffers_[name];                \
      for (auto i = 0; i < num_elements; ++i) {                                \
        l_data[i] = (type_to) inputData[i];                                    \
      }                                                                        \
      return l;                                                                \

    SWITCH(io_data_type_[name])
    #undef OPERATION
  }

  void XlaExecutor::initIO(uint32_t inputsCount, const onnxTensorDescriptor* inputDescriptors,
                           uint32_t  outputsCount, const onnxTensorDescriptor* outputDescriptors) {
    ONNX_ASSERT(num_inputs_ == inputsCount);
    ONNX_ASSERT(num_outputs_ == outputsCount);
    
    #define CHECK_TYPE_AND_SHAPE(VAR)                                         \
      for (auto i = 0; i < num_##VAR##s_; ++i)  {                               \
        const std::string name(VAR##Descriptors[i].name);                       \
        ONNX_ASSERT(io_data_type_.find(name) != io_data_type_.end());           \
        VAR##_buffers_[name] = VAR##Descriptors[i].buffer;                      \
        ONNX_ASSERT(onnxifiToOnnx(VAR##Descriptors[i].dataType) == io_data_type_[name]);        \
        ONNX_ASSERT(VAR##Descriptors[i].dimensions == io_shape_[name].size());  \
        for (auto j = 0; j < io_shape_[name].size(); ++j)  {                    \
          ONNX_ASSERT(io_shape_[name][j].is_int &&                              \
                      io_shape_[name][j].dim == VAR##Descriptors[i].shape[j]);  \
        }                                                                         \
      }                                                                         \

    CHECK_TYPE_AND_SHAPE(input);
    CHECK_TYPE_AND_SHAPE(output);
    #undef CHECK_TYPE_AND_SHAPE
  }

  void XlaExecutor::sendLiterals()  {
    for (auto& l : static_literals_)  {
      auto l_ptr = l.release();
      auto l_data_ptr = xla::TransferParameterToServer(*l_ptr);
      delete l_ptr;
      arguments_.push_back(l_data_ptr.release());
    }
    //WAIT FOR INPUT SYNCHRONIZATION PRIMITIVE
    for (auto it = input_buffers_.begin(); it != input_buffers_.end(); ++it)  {
      auto l = this->inputToLiteral(it->first);
      auto l_ptr = l.release();
      auto l_data_ptr = xla::TransferParameterToServer(*l_ptr);
      delete l_ptr;
      arguments_.push_back(l_data_ptr.release());
    }
  } 

  void XlaExecutor::executeComputation() {
    
    #define OPERATION(type_to, type_from, vec)                                  \
      type_to* destination = (type_to*) output_buffers_[output_names_[i]];      \
      for (auto j = 0; j < num_elements; ++j)  {                                \
        destination[j] = (type_to) outputLiterals[i].data<type_from>()[j];      \
      }                                                                         \
      break; 	 								\

    auto result = xla::ExecuteComputation(computation_, arguments_);
    std::vector<Literal> outputLiterals =  result->DecomposeTuple();
    for (auto i = 0; i < outputLiterals.size(); ++i)  {
      int64_t num_elements = 1;
      for (auto j = 0; j < io_shape_[output_names_[i]].size(); ++j)  {
        num_elements *= io_shape_[output_names_[i]][j].dim;
      }
      SWITCH(io_data_type_[output_names_[i]])
    }
    //SET OUTPUT SYNCHRONIZATION PRIMITIVE
    #undef OPERATION
  }

  XlaTransform::XlaTransform(std::unique_ptr<Graph> ir, const std::string& build_name) :
    builder_(build_name), executor_(new XlaExecutor()), global_param_number_(0) {
    ir_ = std::move(ir);
  }

  XlaTransform::~XlaTransform() {}

  inline Shape XlaTransform::shapeOfValue(const Value* v)  {
    std::vector<int64> sizes;
    for (const Dimension& d : v->sizes()) {
      ONNX_ASSERT(d.is_int);
      sizes.push_back(d.dim);
    }
    return ShapeUtil::MakeShape(onnxToPrimitive(v->elemType()), sizes);
  }

  inline void XlaTransform::registerValueOp(const Value* v, XlaOp& op)  {
    value_to_op_[v] = std::move(op);
  }

  inline void XlaTransform::registerValueOp(const Value* v, XlaOp& op, int index)  {
    value_to_op_[v] = std::move(builder_.GetTupleElement(op, index));
  }

  void XlaTransform::fillIOMetadata()  {
    executor_->num_inputs_ = (uint32_t) ((int64_t) (ir_->inputs().size()) - (int64_t) (ir_->initializers().size()));
    executor_->num_outputs_ = (uint32_t) ir_->outputs().size();
    std::unordered_map<std::string, bool> isInitialized;
    for (const std::string& s : ir_->initializer_names())  {
      isInitialized[s] = true;
    }
    for (auto* v : ir_->inputs())  {
      if (isInitialized.find(v->uniqueName()) == isInitialized.end())  {
        executor_->io_data_type_[v->uniqueName()] = v->elemType();
        executor_->io_shape_[v->uniqueName()] = v->sizes();
      }
    }
    
    for (auto* v : ir_->outputs())  {
      executor_->io_data_type_[v->uniqueName()] = v->elemType();
      executor_->io_shape_[v->uniqueName()] = v->sizes();      
    }
  }

  void XlaTransform::translateGraph() {
    for (const Tensor& t : ir_->initializers())  {
      auto l_ptr = executor_->tensorToLiteral(t);
      executor_->static_literals_.push_back(std::move(l_ptr));
    }
    this->fillIOMetadata();
    for (const Value* v : ir_->inputs())  {
      auto param = builder_.Parameter(global_param_number_++, shapeOfValue(v),
                                      v->uniqueName());
      registerValueOp(v, param);
    }
    for (auto it = ir_->begin(); it != ir_->end(); ++it) {
      if (it->kind() == ONNX_NAMESPACE::Symbol("Relu")) {
        auto input = value_to_op_[it->inputs()[0]];
        auto shape = builder_.GetShape(input);
        TF_CHECK_OK(shape.status());
        auto zero = builder_.ConstantLiteral(*LiteralBase::CreateFromShape(shape.ValueOrDie()));
        auto maximum = builder_.Max(input, zero);
        registerValueOp(it->outputs()[0], maximum);
      } else if (it->kind() == ONNX_NAMESPACE::Symbol("Undefined")) {
        continue;
      }
      else {  
        throw conversion_error("Conversion of node type not supported.");
      }
    }
    std::vector<XlaOp> retValues;
    for(const Value* v : ir_->outputs())  {
      retValues.push_back(value_to_op_[v]);
      executor_->output_names_.push_back(v->uniqueName());
    }
    builder_.Tuple(retValues);
    auto computation_status = builder_.Build();
    TF_CHECK_OK(computation_status.status());
    executor_->computation_ = computation_status.ConsumeValueOrDie();

  }

  XlaExecutor* XlaTransform::executor()  {
    return executor_.release();
  }

  OnnxParser::OnnxParser(const void* serializedModel, size_t serializedModelSize)  {
    ONNX_NAMESPACE::ParseProtoFromBytes(&model_, (const char*) serializedModel, serializedModelSize);
  }

  std::unique_ptr<Graph> OnnxParser::parse()  {
    ONNX_NAMESPACE::shape_inference::InferShapes(model_);
    return ONNX_NAMESPACE::ImportModelProto(model_);
  }  
  #undef SWITCH
}

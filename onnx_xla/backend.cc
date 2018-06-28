#include "onnx_xla/backend.h"

namespace onnx_xla {
  
  std::unique_ptr<Literal> XlaExecutor::initializerToLiteral(const Tensor& t) {

    #define GET_LITERAL(type_from, type_to, vec)                               \
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
      }                                                                        \
      return l;                                                                \

    switch(t.elem_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        GET_LITERAL(float, float, floats)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        GET_LITERAL(complex64, complex64, floats)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:  {
        GET_LITERAL(int32_t, half, int32s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:  {
        GET_LITERAL(int32_t, bool, int32s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:  {
        GET_LITERAL(int32_t, int8, int32s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:  {
        GET_LITERAL(int32_t, int16, int32s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:  {
        GET_LITERAL(int32_t, int32, int32s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:  {
        GET_LITERAL(int32_t, uint8, int32s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
        GET_LITERAL(int32_t, uint16, int32s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
        GET_LITERAL(int64_t, int64, int64s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:  {
        GET_LITERAL(uint64_t, uint32, uint64s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
        GET_LITERAL(uint64_t, uint64, uint64s)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:  {
        GET_LITERAL(double, double, doubles)
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
      case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      default:  {
        throw conversion_error("Tensor not of a data type that can be converted.");
      }
    }
  }

  void XlaExecutor::sendLiterals()  {
    for (auto& l : literals_)  {
      auto l_ptr = l.release();
      auto l_data_ptr = xla::TransferParameterToServer(*l_ptr);
      delete l_ptr; 
      arguments_.push_back(l_data_ptr.release());
    }
  } 

  std::vector<Literal> XlaExecutor::executeComputation() {
    auto result = xla::ExecuteComputation(computation_, arguments_);
    return result->DecomposeTuple();
  }

  XlaTransform::XlaTransform(Graph& ir, const std::string& build_name) :
    ir_(ir), builder_(build_name),
    executor_(new XlaExecutor()), global_param_number_(0) {}

  XlaTransform::~XlaTransform()  {
    delete executor_;
  }

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

  void XlaTransform::translateGraph() {
    for (const Tensor& t : ir_.initializers())  {
      auto l_ptr = executor_->initializerToLiteral(t);
      executor_->literals_.push_back(std::move(l_ptr));
    }

    for (const Value* v : ir_.inputs())  {
      auto param = builder_.Parameter(global_param_number_++, shapeOfValue(v),
                                      v->uniqueName());
      registerValueOp(v, param);
    }
    for (auto it = ir_.begin(); it != ir_.end(); ++it) {
      if (it->kind() == ONNX_NAMESPACE::Symbol("Relu")) {
        auto input = value_to_op_[it->inputs()[0]];
        auto shape = builder_.GetShape(input);
        TF_CHECK_OK(shape.status());
        auto zero = builder_.ConstantLiteral(*LiteralBase::CreateFromShape(shape.ValueOrDie()));
        auto maximum = builder_.Max(input, zero);
        registerValueOp(it->outputs()[0], maximum);
      }  else {
         throw conversion_error("Conversion of node type not supported.");
      }
    }
    std::vector<XlaOp> retValues;
    for(const Value* v : ir_.outputs())  {
      retValues.push_back(value_to_op_[v]);
    }
    builder_.Tuple(retValues);
  
    auto computation_status = builder_.Build();
    TF_CHECK_OK(computation_status.status());
    executor_->computation_ = computation_status.ConsumeValueOrDie();
  }

  XlaExecutor* XlaTransform::executor()  {
    return executor_;
  }
}

#include "onnx_xla/backend.h"

namespace onnx_xla {
  using xla::Literal;
  using xla::ShapeUtil;
  using xla::Shape;
  using xla::primitive_util::NativeToPrimitiveType;
  using xla::XlaOp;
  using xla::GlobalData;
  using xla::LiteralBase;

  using ONNX_NAMESPACE::Tensor;
  using ONNX_NAMESPACE::Value;
  using ONNX_NAMESPACE::Dimension;
 
  std::unique_ptr<Literal> XlaTransform::initializerToLiteral(const Tensor& t) {

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

  void XlaTransform::intializersToLiterals()  {
    for (const Tensor& t : ir_.initializers())  {
      auto l_ptr = initializerToLiteral(t);
      literals_.push_back(std::move(l_ptr));
    }
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
    for (auto it = ir_.begin(); it != ir_.end(); ++it) {
      if (it->kind() == ONNX_NAMESPACE::kParam) {
        for (const Value* v : it->outputs())  {
          auto param = builder_.Parameter(global_param_number++, shapeOfValue(v),
                                          it->inputs()[0]->uniqueName());
          registerValueOp(v, param);
        }
      }
      else if (it->kind() == ONNX_NAMESPACE::Symbol("Relu")) {
        auto input = value_to_op_[it->inputs()[0]];
        auto shape = builder_.GetShape(input);
        TF_CHECK_OK(shape.status());
        auto zero = builder_.ConstantLiteral(*LiteralBase::CreateFromShape(shape.ValueOrDie()));
        auto maximum = builder_.Max(input, zero);
        registerValueOp(it->outputs()[0], maximum);
      } else if(it->kind() == ONNX_NAMESPACE::kReturn) {
          std::vector<XlaOp> retValues;
          for(const Value* v : it->inputs())  {
            retValues.push_back(value_to_op_[v]);
          }
          builder_.Tuple(retValues);
      } else {
         throw conversion_error("Conversion of node type not supported.");
      }
    }
  }

  std::vector<Literal> XlaTransform::executeComputation() {
    auto computation_status = builder_.Build();
    TF_CHECK_OK(computation_status.status());
    auto computation = computation_status.ConsumeValueOrDie();
    std::vector<GlobalData*> arguments;
    for (auto& l : literals_)  {
      arguments.push_back(xla::TransferParameterToServer(*l.release()).release());
    }
    auto result = xla::ExecuteComputation(computation, arguments);
    return result->DecomposeTuple();
  }  
}

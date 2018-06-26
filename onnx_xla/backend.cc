#pragma once

#include "onnx-xla/onnx_xla/backend.h"
#include "tensorflow/compiler/xla/types.h"

namespace onnx-xla {
  std::unique_ptr<Literal> XlaTransform::initializerToLiteral(const Tensor& t) {

    #define GET_LITERAL(type_from, type_to, vec)                               \
      type_from* t_data;                                                       \
      if (t.is_raw_data())  {                                                  \
        t_data = (type_from*) t.raw().c_str();                                 \
      } else {                                                                 \
        t_data = (typeofstorage*) t.vec().data();                              \
      }                                                                        \
                                                                               \
      auto l = make_unique<Literal>(                                           \
              ShapeUtil::MakeShape(NativeToPrimitiveType<type_to>, t.sizes()));\
      int64_t num_elements = std::accumulate(t.begin(), t.end(), (int64_t) 1,  \
                                             std::multiplies<int64_t>());      \
      tensorflow::gtl::MutableArraySlice<type_to> l_data = l->data<type_to>(); \
      for (auto i = 0; i < num_elements; ++i) {                                \
        l_data[i] = (type_to) t_data[i];                                       \
      }                                                                        \
      return l;                                                                \

    switch(t.data_type()) {
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
        throw conversion_error("Tensor not of a data type that can be converted.")
      }
    }
  }

  void XlaTransform::intializersToLiterals()  {
    for (const& Tensor t : ir_.initializers())  {
      auto l_ptr = initializerToLiteral(t);
      literals.push_back(std::move(l_ptr));
    }
  }

  inline Shape XlaTransform::shapeOfValue(const Value* v)  {
    vector<int64> sizes;
    for (const Dimension& d : v->sizes()) {
      ONNX_ASSERT(d.is_int);
      sizes.push_back(d.dim());
    }
    return ShapeUtil::MakeShape(onnxToPrimitive(v->elem_type()), sizes);
  }

  inline void XlaTransform::registerValueOp(const Value* v, XlaOp* op)  {
    value_to_op_[v] = op;
  }

  inline void XlaTransform::registerValueOp(const Value* v, XlaOp* op, int index)  {
    value_to_op_[v] = &(builder_.getTupleElement(*op, index));
  }

  void XlaTransform::buildComputation() {
    for (const auto it = ir_.cbegin(); it != ir_.cend(); ++it) {
      switch(it->kind())  {
        case kParam:  {
          for (const Value* v : it->outputs())  {
            auto param = builder_.Parameter(global_param_number++, shapeOfValue(v),
                                            it->inputs()[0]->uniqueName()));
            registerValueOp(v, &param);
          }
        }
        case "Relu": {
          auto input_ptr = value_to_op_[it->inputs()[0]];
          auto shape = builder.GetShape(*input_ptr);
          TF_CHECK_OK(shape.status());
          auto zero = builder_.ConstantLiteral(LiteralBase::CreateFromShape(
                                               shape.ValueOrDie());
          auto maximum = builder.Max(*input_ptr, zero);
          registerValueOp(it->outputs()[0], maximum);
        }
        case kReturn: {
          std::vector<XlaOp> retValues;
          for(const Value* v : it->inputs())  {
            retValues.push_back(value_to_op_[v]);
          }
          builder_.Tuple(retValues);
        }
        default:
          throw conversion_error("Conversion of node type not supported.");
      }
    }
    auto computation_status = builder_.Build();
    TF_CHECK_OK(computation_status.status());
    computation_ = computation_status.ConsumeValueOrDie();
}

std::vector<Literal> executeComputation() {
  ONNX_ASSERT(computation_ != NULL);
  std::vector<GlobalData*> arguments;
  for (auto l : literals_)  {
    data_vector.push_back(*TransferParameterToServer(*l.release()).release());
  }
  auto result = ExecuteComputation(computation_, arguments);
  return result->DecomposeTuple();
}

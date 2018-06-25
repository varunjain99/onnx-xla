#pragma once

#include "onnx-xla/onnx_xla/backend.h"
#include "tensorflow/compiler/xla/types.h"

namespace onnx-xla {
  Literal* XlaTransform::initializerToLiteral(const Tensor& t)  {

    #define GET_LITERAL(type_from, type_to, vec)                               \
      type_from* t_data;                                                       \
      if (t.is_raw_data())  {                                                  \
        t_data = (type_from*) t.raw().c_str();                                 \
      } else {                                                                 \
        t_data = (typeofstorage*) t.vec().data();                              \
      }                                                                        \
                                                                               \
      Literal* l = new Literal(ShapeUtil::MakeShape(                           \
                               NativeToPrimitiveType<type_to>, t.sizes()));    \
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
      case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:  {
        return NULL;
      }
    }
  }

  bool XlaTransform::intializersToLiterals()  {
    for (const& Tensor t : ir_.initializers())  {
      auto l_ptr = initializerToLiteral(t);
      if (l_ptr != NULL)  {
        literals.push_back(l_ptr);
      } else {
        return false;
      }
    }
    return true;
  }

  inline void XlaTransform::registerValueTypeAndShape(const Value* v)  {
    value_to_primitive_type_[v] = onnxToPrimitive(v->elem_type());
    vector<int64> sizes;
    for (const Dimension& d : v->sizes()) {
      if (d.is_int) {
        sizes.push_back(d.dim());
      } else {
        //error
      }
    }
    value_to_shape_[v] = ShapeUtil::MakeShape(value_to_primitive_type_[v], sizes);
  }

  inline void XlaTransform::registerValueOp(const Value* v, XlaOp* op)  {
    value_to_op_[v] = op;
  }

  inline void XlaTransform::registerValueOp(const Value* v, XlaOp* op, int index)  {
    value_to_op_[v] = &(builder_.getTupleElement(*op, index));
  }

  void XlaTransform::computeBuild() {
    for (const auto it = ir_.cbegin(); it != ir_.cend(); ++it) {
      switch(it->kind())  {
        case kParam:  {
          for (const Value* v : it->outputs())  {
            registerValueTypeAndShape(v);
            auto param = builder_.Parameter(global_param_number++, value_to_shape_[v],
                                            it->inputs()[0]->uniqueName()));
            registerValueOp(v, &param);
          }
        }
        case "Relu": {
          auto input_ptr = value_to_op_[it->inputs()[0]];
          auto zero = builder_.ConstantLiteral(LiteralBase::CreateFromShape(
                                               value_to_shape_[it->inputs()[0]]));
          auto maximum = builder.Max(*input_ptr, zero);
          registerValue(it->outputs()[0], maximum);
          registerValueTypeAndShape(it->outputs()[0]);
        }
        case kReturn: {

        }
        default:
          throw("Onnx operation not yet supported");
      }


    }

    auto computation_status = builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
}

#include "onnx_xla/types.h"

namespace onnx_xla  {
   xla::PrimitiveType onnxToPrimitive(ONNX_NAMESPACE::TensorProto_DataType data_type)  {
      switch(data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        return xla::F32;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        return xla::C64;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:  {
        return xla::F16;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:  {
        return xla::PRED;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:  {
        return xla::S8;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:  {
        return xla::S16;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:  {
        return xla::S32;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:  {
        return xla::U8;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
        return xla::U16;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
        return xla::S64;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:  {
        return xla::U32;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
        return xla::U64;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:  {
        return xla::F64;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
      case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: {
        throw("Not supported");
      }
    }
  }
}

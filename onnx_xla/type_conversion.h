#pragma once

#include "tensorflow/compiler/xla/types.h"
#include "onnx/onnx_pb.h"

using ::xla;
using ::ONNX_NAMESPACE;
namespace onnx-xla  {
  PrimitiveType onnxToPrimitive(ONNX_NAMESPACE::TensorProto_DataType)  {
      switch(t.data_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        return F32;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        return C64;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:  {
        return F16;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:  {
	return PRED;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:  {
	return S8;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:  {
        return S16;      
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:  {
        return S32;      
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:  {
        return U8;      
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
        return U16;      
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
        return S64;      
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:  {
        return U32;      
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
        return U64;      
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:  {
        return F64;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
      case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: {
        throw("Not supported");
      }
    }
  }

}

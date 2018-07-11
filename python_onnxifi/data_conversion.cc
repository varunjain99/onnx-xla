#include "python_onnxifi/data_conversion.h"

#include <algorithm>
#include <functional>









namespace py = pybind11;

void releaseTensorDescriptor(onnxTensorDescriptor& t)  {
  auto num_elements = std::accumulate(t.shape, t.shape + t.dimensions, 1, std::multiplies<size_t>());
  std::cout << "SHAPE: ";
  for (auto i = 0; i < t.dimensions; ++i)  {
    std::cout << t.shape[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "DATA: ";
  for (auto i = 0; i < num_elements; ++i)  {
    std::cout << *((int32_t*) t.buffer + i) << " ";
  }
  std::cout << std::endl;
  delete [] t.shape;
  delete [] reinterpret_cast<char*>(t.buffer);
}

//TODO: Different memory types
template<typename py_type, typename onnx_type>
void fillTensorDescriptor(onnxTensorDescriptor& t, const py::buffer_info& arrayInfo,
                          const ONNX_NAMESPACE::TensorProto_DataType dataType,
                          const std::string& name)  {
  t.name = name.c_str();
  t.dataType = dataType;
  t.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  t.dimensions = arrayInfo.ndim;
  auto shape = new uint64_t[arrayInfo.shape.size()];
  std::copy(arrayInfo.shape.begin(), arrayInfo.shape.end(), shape);
  t.shape = shape;
  auto num_elements = std::accumulate(arrayInfo.shape.begin(), arrayInfo.shape.end(), 1, std::multiplies<size_t>());
  onnx_type* buffer = reinterpret_cast<onnx_type*> (new char[sizeof(onnx_type)*num_elements]);
  py_type* data = reinterpret_cast<py_type*>(arrayInfo.ptr);
  std::copy(data, data + num_elements, buffer);
  t.buffer = reinterpret_cast<onnxPointer>(buffer);
}

void numpyToOnnxTensorDescriptor(onnxTensorDescriptor& t, py::array& py_array, const std::string& name)  {
  auto arrayInfo = py_array.request();
  if (arrayInfo.format == "?")  {
    fillTensorDescriptor<bool, int32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_BOOL, name);
  } else if (arrayInfo.format == "b")  {
    fillTensorDescriptor<int8_t, int32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT8, name);
  } else if (arrayInfo.format == "h")  {
    fillTensorDescriptor<int16_t, int32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT16, name);
  } else if (arrayInfo.format == "i")  {
    fillTensorDescriptor<int32_t, int32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT32, name);
  } else if (arrayInfo.format == "l")  {
    fillTensorDescriptor<int64_t, int64_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT64, name);
  } else if (arrayInfo.format == "B")  {
    fillTensorDescriptor<uint8_t, int32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT8, name);
  } else if (arrayInfo.format == "H")  {
    fillTensorDescriptor<uint16_t, int32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT16, name);
  } else if (arrayInfo.format == "I")  {
    fillTensorDescriptor<uint32_t, uint64_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT32, name);
  } else if (arrayInfo.format == "L")  {
    fillTensorDescriptor<uint64_t, uint64_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT64, name);
  } else if (arrayInfo.format == "e")  {
    //TODO FLOAT16
  } else if (arrayInfo.format == "f")  {
    fillTensorDescriptor<float, float>(t, arrayInfo,
             ONNX_NAMESPACE::TensorProto_DataType_FLOAT, name);
  } else if (arrayInfo.format == "d")  {
    fillTensorDescriptor<double, double>(t, arrayInfo,
             ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, name);
  } else if (arrayInfo.format == "Zf")  {
    fillTensorDescriptor<std::complex<float>, std::complex<float>>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64, name);
  } else if (arrayInfo.format == "Zd")  {
    fillTensorDescriptor<std::complex<double>, std::complex<double>>(t, arrayInfo,
             ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128, name);
  } else {
    throw std::runtime_error("Unsupported numpy data type");
  }
}

void test(py::array py_array, std::string name)  {
  onnxTensorDescriptor t;
  numpyToOnnxTensorDescriptor(t, py_array, name);
  releaseTensorDescriptor(t);
}

PYBIND11_MODULE(python_onnxifi, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test", &test, "Analyze numpy");
}

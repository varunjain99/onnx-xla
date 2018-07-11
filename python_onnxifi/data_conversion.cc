#include "python_onnxifi/data_conversion.h"

#include <algorithm>
#include <functional>
#include <complex>
namespace py = pybind11;

DataConversion::DataConversion() {}

DataConversion::~DataConversion()  {
  releaseDescriptors(weight_descriptors_);
  releaseDescriptors(input_descriptors_);
  releaseDescriptors(output_descriptors_);
}

void DataConversion::makeWeightDescriptors(py::dict& numpyArrays)  {
  makeDescriptorsFromNumpy(numpyArrays, weight_descriptors_);
}

void DataConversion::updateInputDescriptors(py::dict& numpyArrays)  {
  releaseDescriptors(input_descriptors_);
  makeDescriptorsFromNumpy(numpyArrays, input_descriptors_);
}

//TODO: This should be moved to onnx repository: onnx/common/tensor.h
//Proposal: The dispatch switches off tensor type to execute a
//templated function whose first type is the onnx storage type for one element, 
//and the second type is the natural type with size in the case name.
#define DISPATCH_OVER_NUMERIC_DATA_TYPE(data_type, op_template, ...)             \
switch(data_type) {                                                     	 \
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {                    	 \
    op_template<float, float>(__VA_ARGS__);                        	         \
    break;									 \
  } 	                   							 \
  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {			 \
    op_template<std::complex<float>, std::complex<float>>(__VA_ARGS__);          \
    break;									 \
  }										 \
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:  { 			 \
   /*TODO:op_template<int32_t, half>(__VA_ARGS__);*/                       	 \
    break;									 \
  }				 						 \
  case ONNX_NAMESPACE::TensorProto_DataType_BOOL:  { 				 \
    op_template<int32_t, bool>(__VA_ARGS__);          				 \
    break;									 \
  } 										 \
  case ONNX_NAMESPACE::TensorProto_DataType_INT8:  {				 \
    op_template<int32_t, int8_t>(__VA_ARGS__);                     		 \
    break;									 \
  } 	   	      								 \
  case ONNX_NAMESPACE::TensorProto_DataType_INT16:  {				 \
    op_template<int32_t, int16_t>(__VA_ARGS__);          			 \
    break;									 \
  } 										 \
  case ONNX_NAMESPACE::TensorProto_DataType_INT32:  { 				 \
    op_template<int32_t, int32_t>(__VA_ARGS__);          			 \
    break;									 \
  } 			       							 \
  case ONNX_NAMESPACE::TensorProto_DataType_UINT8:  {				 \
    op_template<int32_t, uint8_t>(__VA_ARGS__);          			 \
    break;									 \
  } 		 								 \
  case ONNX_NAMESPACE::TensorProto_DataType_UINT16: { 				 \
    op_template<int32_t, uint16_t>(__VA_ARGS__);          		    	 \
    break;									 \
  } 									         \
  case ONNX_NAMESPACE::TensorProto_DataType_INT64: {				 \
    op_template<int64_t, int64_t>(__VA_ARGS__);                               	 \
    break;                                                               	 \
  }										 \
  case ONNX_NAMESPACE::TensorProto_DataType_UINT32:  {				 \
    op_template<uint64_t, uint32_t>(__VA_ARGS__);                                \
    break;                                                                       \
  }     									 \
  case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {			 	 \
    op_template<uint64_t, uint64_t>( __VA_ARGS__);                            	 \
    break;							        	 \
  }   						                        	 \
  case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {                   	 \
   op_template<double, double>(__VA_ARGS__);                                 	 \
   break;                                                                        \
  }										 \
  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {     			 \
   op_template<std::complex<double>, std::complex<double>>(__VA_ARGS__);         \
   break;									 \
  }										 \
  case ONNX_NAMESPACE::TensorProto_DataType_STRING: 				 \
  case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: {			 \
   throw std::runtime_error("Dispatch received non-numeric data type");          \
  }										 \
}										 \

template<typename onnx_type, typename unused>
void DataConversion::getBufferSize(uint64_t& buffer_size, const uint64_t* shape, uint32_t dimensions)  {
  auto numElements = std::accumulate(shape, shape + dimensions,
                                    (uint64_t) 1, std::multiplies<uint64_t>());
  buffer_size = sizeof(onnx_type) * numElements;
}

void DataConversion::updateOutputDescriptors(ModelProto& model)  {
  releaseDescriptors(output_descriptors_);
  ::ONNX_NAMESPACE::shape_inference::InferShapes(model);
  const auto& outputs = model.graph().output();
  for (const auto& vi : outputs)  {
    onnxTensorDescriptor outputDescriptor;
    char* name = new char[vi.name().size() + 1];
    strcpy(name, vi.name().c_str());
    outputDescriptor.name = name;
    outputDescriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU; //TODO: Expand memory types?
    if (!vi.type().tensor_type().has_elem_type())  {
      throw std::runtime_error("Non-static ModelProto: Output data type not found");
    }
    outputDescriptor.dataType = vi.type().tensor_type().elem_type();
    outputDescriptor.dimensions = vi.type().tensor_type().shape().dim_size();
    auto* shape = new uint64_t[outputDescriptor.dimensions];
    for (auto i = 0; i < outputDescriptor.dimensions; ++i)  {
      const auto& dim = vi.type().tensor_type().shape().dim(i);
      if (!dim.has_dim_value())  {
        throw std::runtime_error("Non-static ModelProto: Output shape dimension not found");
      }
      shape[i] = dim.dim_value();
    }
    outputDescriptor.shape = shape;
    uint64_t buffer_size;
    DISPATCH_OVER_NUMERIC_DATA_TYPE(outputDescriptor.dataType, getBufferSize, buffer_size,
                                    outputDescriptor.shape, outputDescriptor.dimensions)
    auto* buffer = new char[buffer_size];  //(Using char so can delete properly w/o warning)
    outputDescriptor.buffer = reinterpret_cast<onnxPointer>(buffer);
    output_descriptors_.push_back(outputDescriptor); 
  } 
}

template<typename onnx_type, typename py_type>
void DataConversion::addNumpyArray(py::dict& numpyArrays, const onnxTensorDescriptor& t)  {
  auto numElements = std::accumulate(t.shape, t.shape + t.dimensions,
                                    (uint64_t) 1, std::multiplies<uint64_t>());
  py_type intermediateBuffer[numElements];
  onnx_type* oldBuffer = reinterpret_cast<onnx_type*>(t.buffer);
  std::copy(oldBuffer, oldBuffer + numElements, intermediateBuffer);
  std::vector<ptrdiff_t> shape(t.shape, t.shape + t.dimensions);
  auto numpyArray = py::array_t<py_type>(shape, reinterpret_cast<py_type*>(intermediateBuffer)); 
  numpyArrays[py::str(std::string(t.name))] = numpyArray;
}

py::dict DataConversion::getNumpyOutputs() const {
  py::dict outputDict;
  getNumpyFromDescriptors(outputDict, output_descriptors_);
  return outputDict;
}

void DataConversion::getNumpyFromDescriptors(py::dict& numpyArrays, const std::vector<onnxTensorDescriptor>& tensorDescriptors)  {
  for (const auto& descriptor : tensorDescriptors)  {
    DISPATCH_OVER_NUMERIC_DATA_TYPE(descriptor.dataType, addNumpyArray, numpyArrays, descriptor)
  } 
}

void DataConversion::makeDescriptorsFromNumpy(py::dict& numpyArrays, std::vector<onnxTensorDescriptor>& tensorDescriptors)  {
  for (const auto& item : numpyArrays)  {
    onnxTensorDescriptor t;
    std::string nameString(py::str(item.first));
    char* name = new char[nameString.size() + 1];
    strcpy(name, nameString.c_str());
    py::array numpyArray = py::reinterpret_borrow<py::array>(item.second); //TODO: Check if of type py::array
    fillTensorDescriptor(t, numpyArray, name);
    tensorDescriptors.push_back(t);
  }
}

template<typename onnx_type, typename py_type>
void DataConversion::fillTensorDescriptorImpl(onnxTensorDescriptor& t, const py::buffer_info& arrayInfo,
                                              ONNX_NAMESPACE::TensorProto_DataType dataType, const char* name)  {
  t.name = name;
  t.dataType = dataType;
  t.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  t.dimensions = arrayInfo.ndim;
  auto shape = new uint64_t[arrayInfo.shape.size()];
  std::copy(arrayInfo.shape.begin(), arrayInfo.shape.end(), shape);
  t.shape = shape;
  auto numElements = std::accumulate(t.shape, t.shape + t.dimensions, 
                                     (uint64_t) 1, std::multiplies<uint64_t>());
  onnx_type* buffer = reinterpret_cast<onnx_type*> (new char[numElements * sizeof(onnx_type)]);
  py_type* data = reinterpret_cast<py_type*>(arrayInfo.ptr);
  std::copy(data, data + numElements, buffer);
  t.buffer = reinterpret_cast<onnxPointer>(buffer);
}

//TODO: Change from format == letter comparison to using buffer_info format template
void DataConversion::fillTensorDescriptor(onnxTensorDescriptor& t, py::array& py_array, const char* name)  {
  auto arrayInfo = py_array.request();
  if (arrayInfo.format == "?")  {
    fillTensorDescriptorImpl<int32_t, bool>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_BOOL, name);
  } else if (arrayInfo.format == "b")  {
    fillTensorDescriptorImpl<int32_t, int8_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT8, name);
  } else if (arrayInfo.format == "h")  {
    fillTensorDescriptorImpl<int32_t, int16_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT16, name);
  } else if (arrayInfo.format == "i")  {
    fillTensorDescriptorImpl<int32_t, int32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT32, name);
  } else if (arrayInfo.format == "l")  {
    fillTensorDescriptorImpl<int64_t, int64_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_INT64, name);
  } else if (arrayInfo.format == "B")  {
    fillTensorDescriptorImpl<int32_t, uint8_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT8, name);
  } else if (arrayInfo.format == "H")  {
    fillTensorDescriptorImpl<int32_t, uint16_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT16, name);
  } else if (arrayInfo.format == "I")  {
    fillTensorDescriptorImpl<uint64_t, uint32_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT32, name);
  } else if (arrayInfo.format == "L")  {
    fillTensorDescriptorImpl<uint64_t, uint64_t>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_UINT64, name);
  } else if (arrayInfo.format == "e")  {
    //TODO: fillTensorDescriptorImpl<int32_t, half>(t, arrayInfo,
    //         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, name);
  } else if (arrayInfo.format == "f")  {
    fillTensorDescriptorImpl<float, float>(t, arrayInfo,
             ONNX_NAMESPACE::TensorProto_DataType_FLOAT, name);
  } else if (arrayInfo.format == "d")  {
    fillTensorDescriptorImpl<double, double>(t, arrayInfo,
             ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, name);
  } else if (arrayInfo.format == "Zf")  {
    fillTensorDescriptorImpl<std::complex<float>, std::complex<float>>(t, arrayInfo, 
             ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64, name);
  } else if (arrayInfo.format == "Zd")  {
    fillTensorDescriptorImpl<std::complex<double>, std::complex<double>>(t, arrayInfo,
             ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128, name);
  } else {
    throw std::runtime_error("Unsupported numpy data type");
  }
}

void DataConversion::releaseDescriptors(std::vector<onnxTensorDescriptor>& tensorDescriptors)  {
  for (auto& descriptor : tensorDescriptors)  {
    delete [] descriptor.name;
    delete [] descriptor.shape;
    delete [] reinterpret_cast<char*>(descriptor.buffer);
  }
  tensorDescriptors.clear();
}


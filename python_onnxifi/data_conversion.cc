#include "python_onnxifi/data_conversion.h"

#include <algorithm>
#include <complex>
#include <functional>
namespace py = pybind11;

DescriptorData::DescriptorData() {}

DescriptorData::DescriptorData(DescriptorData&& d) noexcept {
  name = std::move(d.name);
  buffer = std::move(d.buffer);
  shape = std::move(d.shape);
  descriptor.name = name.c_str();
  descriptor.dimensions = shape.size();
  descriptor.shape = shape.data();
  descriptor.buffer = reinterpret_cast<onnxPointer>(buffer.data());
  descriptor.memoryType = d.descriptor.memoryType;
  descriptor.dataType = d.descriptor.dataType;
}

DescriptorData::DescriptorData(const DescriptorData& d) {
  name = d.name;
  buffer = d.buffer;
  shape = d.shape;
  descriptor.name = name.c_str();
  descriptor.dimensions = shape.size();
  descriptor.shape = shape.data();
  descriptor.buffer = reinterpret_cast<onnxPointer>(buffer.data());
  descriptor.memoryType = d.descriptor.memoryType;
  descriptor.dataType = d.descriptor.dataType;
}

DataConversion::DataConversion() : input_descriptors_data_(), output_descriptors_data_(), weight_descriptors_data_(){}

DataConversion::~DataConversion() {}

void DataConversion::makeWeightDescriptorsData(py::dict& numpyArrays) {
  makeDescriptorsDataFromNumpy(numpyArrays, weight_descriptors_data_);
}

void DataConversion::updateInputDescriptorsData(py::dict& numpyArrays) {
  input_descriptors_data_.clear();
  makeDescriptorsDataFromNumpy(numpyArrays, input_descriptors_data_);
}

// TODO: This should be moved to onnx repository: onnx/common/tensor.h
// Proposal: The dispatch switches off tensor type to execute a
// templated function whose first type is the onnx storage type for one ddent,
// and the second type is the natural type with size in the case name.
#define DISPATCH_OVER_NUMERIC_DATA_TYPE(data_type, op_template, ...)         \
  switch (data_type) {                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {                       \
      op_template<float, float>(__VA_ARGS__);                                \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {                   \
      op_template<std::complex<float>, std::complex<float> >(__VA_ARGS__);   \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {                     \
      /*TODO:op_template<int32_t, half>(__VA_ARGS__);*/                      \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {                        \
      op_template<int32_t, bool>(__VA_ARGS__);                               \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {                        \
      op_template<int32_t, int8_t>(__VA_ARGS__);                             \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {                       \
      op_template<int32_t, int16_t>(__VA_ARGS__);                            \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {                       \
      op_template<int32_t, int32_t>(__VA_ARGS__);                            \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {                       \
      op_template<int32_t, uint8_t>(__VA_ARGS__);                            \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {                      \
      op_template<int32_t, uint16_t>(__VA_ARGS__);                           \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {                       \
      op_template<int64_t, int64_t>(__VA_ARGS__);                            \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {                      \
      op_template<uint64_t, uint32_t>(__VA_ARGS__);                          \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {                      \
      op_template<uint64_t, uint64_t>(__VA_ARGS__);                          \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {                      \
      op_template<double, double>(__VA_ARGS__);                              \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {                  \
      op_template<std::complex<double>, std::complex<double> >(__VA_ARGS__); \
      break;                                                                 \
    }                                                                        \
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:                        \
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: {                   \
      throw std::runtime_error("Dispatch received non-numeric data type");   \
    }                                                                        \
  }

template <typename onnx_type, typename unused>
void DataConversion::getBufferSize(uint64_t& buffer_size,
                                   const uint64_t* shape,
                                   uint32_t dimensions) {
  auto numElements = std::accumulate(shape, shape + dimensions, 1UL,
                                     std::multiplies<uint64_t>());
  buffer_size = sizeof(onnx_type) * numElements;
}

void DataConversion::updateOutputDescriptorsData(ModelProto& model) {
  output_descriptors_data_.clear();
  ::ONNX_NAMESPACE::shape_inference::InferShapes(model);
  const auto& outputs = model.graph().output();
  for (const auto& vi : outputs) {
    DescriptorData outputData;
    outputData.name = vi.name();
    outputData.descriptor.name = outputData.name.c_str();
    if (!vi.type().tensor_type().has_elem_type()) {
      throw std::runtime_error(
          "Non-static ModelProto: Output data type not found");
    }
    outputData.descriptor.dataType = vi.type().tensor_type().elem_type();
    outputData.descriptor.dimensions =
        vi.type().tensor_type().shape().dim_size();
    for (auto i = 0; i < outputData.descriptor.dimensions; ++i) {
      const auto& dim = vi.type().tensor_type().shape().dim(i);
      if (!dim.has_dim_value()) {
        throw std::runtime_error(
            "Non-static ModelProto: Output shape dimension not found");
      }
      outputData.shape.emplace_back(dim.dim_value());
    }
    outputData.descriptor.shape = outputData.shape.data();
    outputData.descriptor.memoryType =
        ONNXIFI_MEMORY_TYPE_CPU;  // TODO: Expand memory types?
    uint64_t buffer_size;
    DISPATCH_OVER_NUMERIC_DATA_TYPE(
        outputData.descriptor.dataType, getBufferSize, buffer_size,
        outputData.descriptor.shape, outputData.descriptor.dimensions)
    outputData.buffer.resize(buffer_size);
    outputData.descriptor.buffer =
        reinterpret_cast<onnxPointer>(outputData.buffer.data());
    output_descriptors_data_.emplace_back(std::move(outputData));
  }
}

template <typename onnx_type, typename py_type>
void DataConversion::addNumpyArray(py::dict& numpyArrays,
                                   const DescriptorData& dd) {
  auto numElements = std::accumulate(dd.shape.begin(), dd.shape.end(),
                                     1UL, std::multiplies<uint64_t>());
  py_type intermediateBuffer[numElements];
  onnx_type* oldBuffer = reinterpret_cast<onnx_type*>(dd.descriptor.buffer);
  std::copy(oldBuffer, oldBuffer + numElements, intermediateBuffer);
  std::vector<ptrdiff_t> shape(dd.descriptor.shape,
                               dd.descriptor.shape + dd.descriptor.dimensions);
  auto numpyArray = py::array_t<py_type>(shape, intermediateBuffer);
  numpyArrays[py::str(std::string(dd.descriptor.name))] = numpyArray;
}

py::dict DataConversion::getNumpyOutputs() const {
  py::dict outputDict;
  getNumpyFromDescriptorsData(outputDict, output_descriptors_data_);
  return outputDict;
}

void DataConversion::getNumpyFromDescriptorsData(
    py::dict& numpyArrays,
    const std::vector<DescriptorData>& descriptorsData) {
  for (const auto& dd : descriptorsData) {
    DISPATCH_OVER_NUMERIC_DATA_TYPE(dd.descriptor.dataType, addNumpyArray,
                                    numpyArrays, dd)
  }
}

void DataConversion::makeDescriptorsDataFromNumpy(
    py::dict& numpyArrays,
    std::vector<DescriptorData>& descriptorsData) {
  for (const auto& item : numpyArrays) {
    DescriptorData dd;
    py::array numpyArray = py::reinterpret_borrow<py::array>(
        item.second);  // TODO: Check if of type py::array
    fillDescriptorData(dd, numpyArray, std::string(py::str(item.first)));
    descriptorsData.emplace_back(std::move(dd));
  }
}

template <typename onnx_type, typename py_type>
void DataConversion::fillDescriptorDataImpl(
    DescriptorData& dd,
    const py::buffer_info& arrayInfo,
    ONNX_NAMESPACE::TensorProto_DataType dataType,
    const std::string& name) {
  dd.name = name;
  dd.descriptor.name = dd.name.c_str();
  dd.descriptor.dataType = dataType;
  dd.descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  dd.descriptor.dimensions = arrayInfo.ndim;
  dd.shape.resize(dd.descriptor.dimensions);
  std::copy(arrayInfo.shape.begin(), arrayInfo.shape.end(), dd.shape.begin());
  dd.descriptor.shape = dd.shape.data();
  auto numElements = std::accumulate(dd.shape.begin(), dd.shape.end(),
                                     1UL, std::multiplies<uint64_t>());
  dd.buffer.resize(numElements * sizeof(onnx_type));
  onnx_type* buffer = reinterpret_cast<onnx_type*>(dd.buffer.data());
  py_type* data = reinterpret_cast<py_type*>(arrayInfo.ptr);
  std::copy(data, data + numElements, buffer);
  dd.descriptor.buffer = reinterpret_cast<onnxPointer>(buffer);
}

// TODO: Change from format == letter comparison to using buffer_info format
// template
void DataConversion::fillDescriptorData(DescriptorData& dd,
                                        py::array& py_array,
                                        const std::string& name) {
  auto arrayInfo = py_array.request();
  if (arrayInfo.format == "?") {
    fillDescriptorDataImpl<int32_t, bool>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_BOOL, name);
  } else if (arrayInfo.format == "b") {
    fillDescriptorDataImpl<int32_t, int8_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_INT8, name);
  } else if (arrayInfo.format == "h") {
    fillDescriptorDataImpl<int32_t, int16_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_INT16, name);
  } else if (arrayInfo.format == "i") {
    fillDescriptorDataImpl<int32_t, int32_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_INT32, name);
  } else if (arrayInfo.format == "l") {
    fillDescriptorDataImpl<int64_t, int64_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_INT64, name);
  } else if (arrayInfo.format == "B") {
    fillDescriptorDataImpl<int32_t, uint8_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_UINT8, name);
  } else if (arrayInfo.format == "H") {
    fillDescriptorDataImpl<int32_t, uint16_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_UINT16, name);
  } else if (arrayInfo.format == "I") {
    fillDescriptorDataImpl<uint64_t, uint32_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_UINT32, name);
  } else if (arrayInfo.format == "L") {
    fillDescriptorDataImpl<uint64_t, uint64_t>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_UINT64, name);
  } else if (arrayInfo.format == "e") {
    // TODO: fillDescriptorDataImpl<int32_t, half>(dd, arrayInfo,
    //         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, name);
  } else if (arrayInfo.format == "f") {
    fillDescriptorDataImpl<float, float>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, name);
  } else if (arrayInfo.format == "d") {
    fillDescriptorDataImpl<double, double>(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, name);
  } else if (arrayInfo.format == "Zf") {
    fillDescriptorDataImpl<std::complex<float>, std::complex<float> >(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64, name);
  } else if (arrayInfo.format == "Zd") {
    fillDescriptorDataImpl<std::complex<double>, std::complex<double> >(
        dd, arrayInfo, ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128, name);
  } else {
    throw std::runtime_error("Unsupported numpy data type");
  }
}

std::vector<onnxTensorDescriptor> DataConversion::getInputTensorDescriptors() {
  return getTensorDescriptors(input_descriptors_data_);
}

std::vector<onnxTensorDescriptor> DataConversion::getOutputTensorDescriptors() {
  return getTensorDescriptors(output_descriptors_data_);
}

std::vector<onnxTensorDescriptor> DataConversion::getWeightTensorDescriptors() {
  return getTensorDescriptors(weight_descriptors_data_);
}

std::vector<onnxTensorDescriptor> DataConversion::getTensorDescriptors(
    const std::vector<DescriptorData>& descriptorsData) {
  std::vector<onnxTensorDescriptor> tensorDescriptors;
  for (const auto& dd : descriptorsData) {
    tensorDescriptors.emplace_back(dd.descriptor);
  }
  return tensorDescriptors;
}

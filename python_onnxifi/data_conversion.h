#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "onnx/onnxifi.h"
#include "onnx/onnx.pb.h"
#include "onnx/shape_inference/implementation.h"

#include <vector>

namespace py = pybind11;

using ::ONNX_NAMESPACE::ModelProto;
// Utility class to navigate conversions of onnxTensorDescriptor and numpy
// arrays
// NOTE: py::dict is not const anywhere because pybind11 request function is not
class DataConversion final {
 public:
  // Structor to manage onnxTensorDescriptor data for lifetime of the descriptor
  struct DescriptorData {
    DescriptorData() {}
    // Guard against copying large data
    DescriptorData(DescriptorData const&) = delete;
    // Ensure pointers point to beginning of vectors/strings
    DescriptorData(DescriptorData&& d) noexcept {
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

    onnxTensorDescriptor descriptor;
    std::vector<uint64_t> shape;
    std::string name;
    // container for bytes
    std::vector<char> buffer;
  };

  // Constructor creates empty DataConversion object
  // 1 DataConversion object for every onnxGraph or BackendRep
  DataConversion();

  // Release any resources allocated in by object
  ~DataConversion();

  // TODO: Implement some sort of checkNumpyWithModel function that checks if
  // numpy inputs
  // and weights are valid (particularly useful for static environment)

  // Fills up weight_descriptors_data_ vector
  // Only called once, when preparing the model
  void makeWeightDescriptorsData(py::dict& numpyArrays);

  // TODO: If input shape/type have not changed (static environment)
  // updateInputDescriptorsData, updateOutputDescriptorsData, onnxSetGraphIO
  // should not be called
  // Different member functions of this class should be implemented to fill the
  // I/O buffers already allocated

  // Releases any old input descriptors data if they are stored
  // Fills up input_descriptors_data_ with new input descriptors
  void updateInputDescriptorsData(py::dict& numpyArrays);

  // Releases any old output descriptors data if they are stored
  // Fills up output_descriptors_data_ with new output descriptors
  // TODO: Support for dynamic environment - the shape inference should be
  // dependent on current input_descriptors_data_
  //      Currently only static environment support
  void updateOutputDescriptorsData(ModelProto& model);

  // Returns dictionary of of name to numpy array from output_descriptors_data_
  py::dict getNumpyOutputs() const;

  // Input: empty dictionary
  // Output: dictionary full of numpy outputs
  // Convert from output onnxTensorDescriptor to numpy dictionary
  static void getNumpyFromDescriptorsData(
      py::dict& numpyArrays,
      const std::vector<DescriptorData>& descriptorsData);

  // Input: numpy arrays passed in through python interface, empty
  // DescriptorData array
  // Output: descriptorsData is filled with appropriate values
  static void makeDescriptorsDataFromNumpy(
      py::dict& numpyArrays,
      std::vector<DescriptorData>& descriptorsData);

  // Returns vector of just the onnxTensorDescriptor structure, for input,
  // output, and weight
  std::vector<onnxTensorDescriptor> getInputTensorDescriptors();
  std::vector<onnxTensorDescriptor> getOutputTensorDescriptors();
  std::vector<onnxTensorDescriptor> getWeightTensorDescriptors();

  // Helper to above functions, giving onnxTensorDescriptor from DescriptorData
  // vector
  static std::vector<onnxTensorDescriptor> getTensorDescriptors(
      const std::vector<DescriptorData>& descriptorsData);

 private:
  // Helper to updateOutputDescriptorsData
  // Uses onnx_type and number of elements in tensor to store number of bytes
  // needed
  template <typename onnx_type, typename unused>
  static void getBufferSize(uint64_t& buffer_size,
                            const uint64_t* shape,
                            uint32_t dimensions);

  // Helper to getNumpyFromOutputDescriptorsData
  // Converts data in tensor descriptor into numpy array, adding it to the
  // dictionary
  template <typename onnx_type, typename py_type>
  static void addNumpyArray(py::dict& numpyArrays, const DescriptorData& dd);

  // Helpers to makeDescriptorsDataFromNumpy
  // Fills up onnx TensorDescriptor with metadata and data
  template <typename onnx_type, typename py_type>
  static void fillDescriptorDataImpl(
      DescriptorData& dd,
      const py::buffer_info& arrayInfo,
      ONNX_NAMESPACE::TensorProto_DataType dataType,
      const std::string& name);

  // Execute dispatch for fillTensorDescriptorImpl
  static void fillDescriptorData(DescriptorData& dd,
                                 py::array& py_array,
                                 const std::string& name);

  // Vectors of input, output, and weight tensor Descriptors
  std::vector<DescriptorData> input_descriptors_data_;
  std::vector<DescriptorData> output_descriptors_data_;
  std::vector<DescriptorData> weight_descriptors_data_;
};

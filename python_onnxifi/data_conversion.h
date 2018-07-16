#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "onnx/onnxifi.h"
#include "onnx/onnx.pb.h"
#include "onnx/shape_inference/implementation.h"

#include <vector>
#include <unordered_map>

namespace py = pybind11;

//Friend function for testing purposes
py::dict convert(py::dict inputDict);

using ::ONNX_NAMESPACE::ModelProto;
// Utility class to navigate conversions of onnxTensorDescriptor and numpy
// arrays
// NOTE: py::dict is not const anywhere because pybind11 request function is not

// Structor to manage onnxTensorDescriptor data for lifetime of the descriptor
struct DescriptorData {
  DescriptorData();
  DescriptorData(const DescriptorData& d);
  // Ensure pointers point to beginning of vectors/strings
  DescriptorData(DescriptorData&& d) noexcept;

  onnxTensorDescriptor descriptor;
  std::vector<uint64_t> shape;
  std::string name;
  // container for bytes
  std::vector<char> buffer;
};

//TODO: Weight descriptors
//TODO: Support for strides
class DataConversion final {
 public:
  friend py::dict convert(py::dict inputDict);
  // Constructor creates DataConversion object with underlying model
  // Initializes underlying member variables
  // 1 DataConversion object for every onnxGraph or BackendRep
  DataConversion(const std::string& serializedModel);

  // Release any resources allocated in by object
  ~DataConversion();

  // TODO: Implement some sort of checkNumpyWithModel function that checks if
  // numpy inputs
  // and weights are valid (particularly useful for static environment)

  // Returns dictionary of of name to numpy array from output_descriptors_data_
  py::dict getNumpyDictOutputs() const;

  //Fills up the input_descriptors_data_ and output_descriptors_data_ member variables,
  //The boolen value returned indicates whether of not the descriptor values were
  // changed by the call (i.e. whether onnxSetGraphIO needs to be called
  bool updateDescriptors(py::dict& inputs);

  //Return tensor descriptors from the stored member variables
  std::vector<onnxTensorDescriptor> getInputDescriptors() const;
  std::vector<onnxTensorDescriptor> getOutputDescriptors() const;


 private:


  /************************************************************************/
  /*******   MODEL PROTO OUTPUTS   ----> DESCRIPTOR DATA MAP **************/

  // Fills up descriptorsData map based on ModelProto graph outputs
  // Throws exception if cannot infer all necessary info
  static void DescriptorDataMapFromModelProtoOutputs(ModelProto& model, std::unordered_map<std::string, DescriptorData>& descriptorsData);

  // Helper function
  // Uses onnx_type and tensor shape to compute buffer_size bytes
  template <typename onnx_type, typename unused>
  static void getBufferSize(uint64_t& buffer_size,
                            const uint64_t* shape,
                            uint32_t dimensions);


  /************************************************************************/
  /*******   NUMPY ARRAY DICT   ----> DESCRIPTOR DATA MAP   ***************/
  
  // Fills up descriptorData map based on values from numpyArrays dictionary
  static void DescriptorDataMapFromNumpyDict(
      py::dict& numpyArrays,
      std::unordered_map<std::string, DescriptorData>& descriptorsData);


  // Helper function
  // Fills up DescriptorData with metadata and data given type info
  template <typename onnx_type, typename py_type>
  static void fillDescriptorDataImpl(
      DescriptorData& dd,
      const py::buffer_info& arrayInfo,
      ONNX_NAMESPACE::TensorProto_DataType dataType,
      const std::string& name);

  // Helper function 
  // Executes disaptch to fillDescriptorDataImpl
  static void fillDescriptorData(DescriptorData& dd,
                                 py::array& py_array,
                                 const std::string& name);


  /************************************************************************/
  /*******  DESCRIPTOR DATA MAP  ---->  DESCRIPTOR DATA MAP ***************/

  // Fills up numpyArrays dictionary from descriptorsData map
  static void NumpyDictFromDescriptorDataMap(
      py::dict& numpyArrays,
      const std::unordered_map<std::string, DescriptorData>& descriptorsData);

  // Helper function
  // Adds one array to numpyArrays dict, based on the dd DescriptorData object 
  template <typename onnx_type, typename py_type>
  static void addNumpyArray(py::dict& numpyArrays, const DescriptorData& dd);


  /************************************************************************/
  /******  DESCRIPTOR DATA MAP   ----> ONNX TENSOR DESCRIPTOR VECTOR  *****/

  static std::vector<onnxTensorDescriptor> getTensorDescriptors(
      const std::unordered_map<std::string, DescriptorData>& descriptorsData);

  /************************************************************************/
  /*********************PRIVATE MEMBER VARIABLES***************************/

  // Vectors of input, output, and weight tensor Descriptors
  std::unordered_map<std::string, DescriptorData> input_descriptors_data_;
  std::unordered_map<std::string, DescriptorData> output_descriptors_data_;
  std::unordered_map<std::string, DescriptorData> weight_descriptors_data_;

  //ModelProto model the object uses  
  ModelProto model_;
};

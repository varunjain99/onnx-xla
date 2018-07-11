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
//Utility class to navigate conversions of onnxTensorDescriptor and numpy arrays
//NOTE: py::dict is not const anywhere because pybind11 request function is not
class DataConversion final  {
public:
  //Constructor creates empty DataConversion object
  //1 DataConversion object for every onnxGraph or BackendRep
  DataConversion();

  //Release any resources allocated in tensor descriptors
  ~DataConversion();

  //TODO: Implement some sort of checkNumpyWithModel function that checks if numpy inputs 
  //and weights are valid (particularly useful for static environment)
 
  //Fills up weight_descriptors_ vector
  //Only called once, when preparing the model
  //Not necessary to releaseDescriptors (handled by destructor)
  void makeWeightDescriptors(py::dict& numpyArrays);
 
  //TODO: If input shape/type have not changed (static environment)
    // updateInputDescriptors, updateOutputDescriptors, onnxSetGraphIO should not be called
    // Different member functions of this class should be implemented to fill the I/O buffers already allocated

  //Releases any old input descriptors if they are stored
  //Fills up input_descriptors_ with new input descriptors
  //Not necessary to releaseDescriptors (handled by destructor)
  void updateInputDescriptors(py::dict& numpyArrays);      

  //Releases any old output descriptors if they are stored
  //Fills up output_descriptors_ with new output descriptors
  //TODO: Support for dynamic environment - the shape inference should be dependent on current input_descriptors_
  //      Currently only static environment support
  //Not necessary to releaseDescriptors (handled by destructor)
  void updateOutputDescriptors(ModelProto& model);

  //Returns dictionary of of name to numpy array from output_descriptors_
  py::dict getNumpyOutputs() const;

  //Input: empty dictionary
  //Output: dictionary full of numpy outputs
  //Convert from output onnxTensorDescriptor to numpy dictionary
  static void getNumpyFromDescriptors(py::dict& numpyArrays, const std::vector<onnxTensorDescriptor>& tensorDescriptors);

  //Input: numpy arrays passed in through python interface, empty onnxTensorDescriptor array
  //Output: tensorDescriptors is filled with appropriate values
  //Note: If this is directly called, must releaseDescriptors
  static void makeDescriptorsFromNumpy(py::dict& numpyArrays, std::vector<onnxTensorDescriptor>& tensorDescriptors);

  //Releases resources allocated by tensorDescriptors and clears it
  static void releaseDescriptors(std::vector<onnxTensorDescriptor>& tensorDescriptors);
private:
  //Helper to updateOutputDescriptors
  //Uses onnx_type and number of elements in tensor to store number of bytes needed
  template<typename onnx_type, typename unused>
  static void getBufferSize(uint64_t& buffer_size, const uint64_t* shape, uint32_t dimensions);

  //Helper to getNumpyFromOutputDescriptors
  //Converts data in tensor descriptor into numpy array, adding it to the dictionary
  template<typename onnx_type, typename py_type>
  static void addNumpyArray(py::dict& numpyArrays, const onnxTensorDescriptor& t);

  //Helpers to makeDescriptorsFromNumpy
  //Fills up onnx TensorDescriptor with metadata and data
  template<typename onnx_type, typename py_type>
  static void fillTensorDescriptorImpl(onnxTensorDescriptor& t, const py::buffer_info& arrayInfo,
                                       ONNX_NAMESPACE::TensorProto_DataType dataType, const char* name); 

  //Execute dispatch for fillTensorDescriptorImpl
  static void fillTensorDescriptor(onnxTensorDescriptor& t, py::array& py_array, const char* name);

  //Vectors of input, output, and weight tensor Descriptors
  std::vector<onnxTensorDescriptor> input_descriptors_;
  std::vector<onnxTensorDescriptor> output_descriptors_;
  std::vector<onnxTensorDescriptor> weight_descriptors_;
};

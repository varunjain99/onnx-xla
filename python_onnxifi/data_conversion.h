#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "onnx/onnxifi.h"
#include "onnx/onnx.pb.h"
#include "onnx/shape_inference/implementation.h"

#include <vector>

namespace py = pybind11;

//Utility class to navigate conversions of onnxTensorDescriptor and numpy arrays
class DataConversion  {
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
  void makeWeightDescriptors(const py::dict& numpyArrays);
 
  //TODO: If input shape/type have not changed (static environment)
    // updateInputDescriptors, updateOutputDescriptors, onnxSetGraphIO should not be called
    // Different member functions of this class should be implemented to fill the I/O buffers already allocated

  //Releases any old input descriptors if they are stored
  //Fills up input_descriptors_ with new input descriptors
  void updateInputDescriptors(const py::dict& numpyArrays);      

  //Releases any old output descriptors if they are stored
  //Fills up output_descriptors_ with new output descriptors
  //TODO: In dynamic environment, the shape inference will be dependent on current input_descriptors_ 
  void updateOutputDescriptors(ModelProto& model);

  //Input: empty dictionary
  //Output: dictionary full of numpy outputs
  //Convert from output onnxTensorDescriptor to numpy dictionary
  void getNumpyFromOutputDescriptors(py::dict& numpyArrays);

private:
  //Input: numpy arrays passed in through python interface, empty onnxTensorDescriptor array
  //Output: tensorDescriptors is filled with appropriate values
  static void makeDescriptorsFromNumpy(const py::dict& numpyArrays, std::vector<onnxTensorDescriptor>& tensorDescriptors);

  //Releases resources allocated by tensorDescriptors and clears it
  static void releaseDescriptors(std::vector<onnxTensorDescriptor>& tensorDescriptors);

  //Vectors of input, output, and weight tensor Descriptors
  std::vector<onnxTensorDescriptor> input_descriptors_;
  std::vector<onnxTensorDescriptor> output_descriptors_;
  std::vector<onnxTensorDescriptor> weight_descriptors_;
}

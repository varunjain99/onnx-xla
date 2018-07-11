#include "python_onnxifi/data_conversion.h"

py::dict convert(py::dict inputDict)  {
  std::vector<onnxTensorDescriptor> tensorDescriptors;
  DataConversion::makeDescriptorsFromNumpy(inputDict, tensorDescriptors);
  py::dict outputDict;
  DataConversion::getNumpyFromDescriptors(outputDict, tensorDescriptors);
  DataConversion::releaseDescriptors(tensorDescriptors);
  return outputDict;
};

PYBIND11_MODULE(data_conversion_test, m) {
  m.def("convert", &convert, "A function which returns the dictionary it received, testing two functions from DataConversions");
}



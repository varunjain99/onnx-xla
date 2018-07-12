#include "python_onnxifi/data_conversion.h"

py::dict convert(py::dict inputDict)  {
  std::vector<DataConversion::DescriptorData> descriptorsData;
  DataConversion::makeDescriptorsDataFromNumpy(inputDict, descriptorsData);
  py::dict outputDict;
  DataConversion::getNumpyFromDescriptorsData(outputDict, descriptorsData);
  return outputDict;
};

PYBIND11_MODULE(data_conversion_test, m) {
  m.def("convert", &convert, "A function which returns the dictionary it received, testing two functions from DataConversions");
}



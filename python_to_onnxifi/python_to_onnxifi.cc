#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string.h>
#include <google/protobuf/message_lite.h>
#include <unordered_map>
#include <utility>
#include <cstdio>
#include "onnx/onnxifi.h"
#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"

namespace py = pybind11;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::NodeProto;

struct DeviceIDs  {
public:
  //Get all backendIDs from ONNXIFI interface
  //Fill device_to_onnxBackendID map
  DeviceIDs() : ids_{nullptr}, num_backends_(0)  {
    if (onnxGetBackendIDs(ids_, &num_backends_) != ONNXIFI_STATUS_FALLBACK) {
      throw std::runtime_error("Internal error: onnxGetBackendIDs failed (expected ONNXIFI_STATUS_FALLBACK)");
    }
    ids_ = new onnxBackendID[num_backends_];
    if (onnxGetBackendIDs(ids_, &num_backends_) != ONNXIFI_STATUS_SUCCESS) {
      throw std::runtime_error("Internal error: onnxGetBackendIDs failed (expected ONNXIFI_STATUS_SUCCESS)");
    }

    for (auto i = 0; i < num_backends_; ++i)  {
      onnxEnum deviceType;
      size_t infoSize = sizeof(onnxEnum);
      if (onnxGetBackendInfo(ids_[i], ONNXIFI_BACKEND_DEVICE_TYPE, 
                            &deviceType, &infoSize) != ONNXIFI_STATUS_SUCCESS)  {
        throw std::runtime_error("Internal Error: onnxGetBackendInfo failed (expected ONNXIFI_STATUS_SUCCESS)");
      }
      if (deviceType_to_string_.find(deviceType) == deviceType_to_string_.end())  {
        throw std::runtime_error("Internal Error: onnxGetBackendInfo returned an invalid device type");
      }
      device_to_onnxBackendID_[deviceType].push_back(ids_[i]);
    }
  }

  //Returns a vector of pairs - one pair for each backendID
    //First element of the pair is a device handle used from python to run on a device
    //Second element is a device description from the ONNXIFI interface
    //Example of the pair: ("GPU:1", "gpu description")
  std::vector<std::pair<std::string, std::string>> getDeviceInfo()  {
    std::vector<std::pair<std::string, std::string>> deviceInfoVector;
    for (auto it : device_to_onnxBackendID_)  {
      auto& deviceType = it.first;
      auto& deviceList = it.second;
      for (auto i = 0; i < deviceList.size(); ++i)  {
        size_t deviceInfoSize = 0;
        if (onnxGetBackendInfo(deviceList[i], ONNXIFI_BACKEND_DEVICE,
                               NULL, &deviceInfoSize) != ONNXIFI_STATUS_FALLBACK)  {
          throw std::runtime_error("Internal Error: onnxGetBackendInfo failed (expected ONNXIFI_STATUS_FALLBACK)");
        }
        char* deviceInfo = new char[deviceInfoSize];
        if (onnxGetBackendInfo(deviceList[i], ONNXIFI_BACKEND_DEVICE,
                               deviceInfo, &deviceInfoSize) != ONNXIFI_STATUS_SUCCESS)  {
          throw std::runtime_error("Internal Error: onnxGetBackendInfo failed (expected ONNXIFI_STATUS_SUCCESS)");
        }
        auto it = deviceType_to_string_.find(deviceType);
        if (it == deviceType_to_string_.end())  {
          throw std::runtime_error("Internal Error: onnxBackendInfo returned unexpected device type");
        }
        std::string deviceHandle(it->second + std::string(":") + std::to_string(i + 1));
        deviceInfoVector.push_back(std::pair<std::string, std::string>(deviceHandle, deviceInfo));
      }
    }
    return deviceInfoVector;
  }

  //parses deviceHandle string (e.g. CPU:1 or GPU) and returns onnxBackendID
  //if valid device, otherwise NULL;
  onnxBackendID getBackendID(const std::string& deviceHandle)  {
    std::string deviceString;
    size_t deviceID = 0;
    size_t position = deviceHandle.find(':');
    if (position != std::string::npos)  {
      std::sscanf(deviceHandle.c_str() + position + 1, "%zu", &deviceID);
      deviceString = deviceHandle.substr(0, position);
    }
    auto it = string_to_deviceType_.find(deviceString);
    if (it == string_to_deviceType_.end())  {
      return NULL;
    }
    auto deviceType = it->second;
    //Treating ID of 0 as 1 and bringing 1-indexing to 0-indexing
    if (deviceID > 0) {
      deviceID--;
    }
    if (deviceID < 0 || deviceID >= device_to_onnxBackendID_[deviceType].size())  {
      return NULL;
    }
    return device_to_onnxBackendID_[deviceType][deviceID];
  }

  //Input is a string device handle 
    //(e.g. "CPU" or "GPU:1" or "CPU")
    //ID of 0 can mean any device of the given type (default to ID 1 for now)
  //Initializes backend if not initialized
  //Updates initialized_ map
  //Returns onnxBackend pointer (NULL if input string is not a valid device)
  onnxBackend prepDevice(const std::string& deviceHandle)  {
    auto id = this->getBackendID(deviceHandle);
    if (!id)  {
      return NULL;
    }
    if (initialized_.find(id) == initialized_.end())  {
      onnxBackend backend;
      if (onnxInitBackend(id, NULL, &backend) != ONNXIFI_STATUS_SUCCESS)  {
        throw("Internal error: onnxInitBackend failed (expected ONNXIFI_STATUS_SUCCESS)");
      }
      initialized_[id] = backend;
    }
    return initialized_[id];
  }

  //Release initialized backends
  //Release all backendIDs
  ~DeviceIDs()  {
    for (auto backend : initialized_) {
      if (onnxReleaseBackend(backend.second) != ONNXIFI_STATUS_SUCCESS)  {
        throw("Internal error: onnxReleaseBackend failed (expected ONNXIFI_STATUS_SUCCESS)");
      }
    }
    if (!ids_) {
      for (auto i = 0; i < num_backends_; ++i)  {
        if (onnxReleaseBackendID(ids_[i]) != ONNXIFI_STATUS_SUCCESS)  {
          throw("Internal error: onnxReleaseBackendID failed (expected ONNXIFI_STATUS_SUCCESS)");
        }
      }
      delete [] ids_;
    }
  }

  //Convenience maps to map betweeen onnxEnum and string
  //  python user will input the string
  //  C++ internals and ONNXIFI represent as onnxEnum
  static const std::unordered_map<onnxEnum, std::string> deviceType_to_string_; 

  static const std::unordered_map<std::string, onnxEnum> string_to_deviceType_; 
private: 
  //Array of all backendIDs
  onnxBackendID* ids_;
  size_t num_backends_;

  //device_to_onnxBackendID[deviceType][deviceID - 1] will give corresponding onnxBackendID
  //1-indexed to conform to onnx/onnx/backend/base.py
  //deviceID of 0 refers to any onnxBackendID of that deviceType
  std::unordered_map<onnxEnum, std::vector<onnxBackendID>> device_to_onnxBackendID_;

  //If the backendID has been initialized, will map to corresponding backendID pointer
  //Otherwise, not present
  std::unordered_map<onnxBackendID, onnxBackend> initialized_;
};

const std::unordered_map<onnxEnum, std::string> DeviceIDs::deviceType_to_string_ = {{ONNXIFI_DEVICE_TYPE_NPU, "NPU"},
                                                                                    {ONNXIFI_DEVICE_TYPE_DSP, "DSP"},
                                                                                    {ONNXIFI_DEVICE_TYPE_GPU, "GPU"},
                                                                                    {ONNXIFI_DEVICE_TYPE_CPU, "CPU"},
                                                                                    {ONNXIFI_DEVICE_TYPE_FPGA, "FPGA"},
                                                                                    {ONNXIFI_DEVICE_TYPE_HETEROGENEOUS, "HETEROGENEOUS"}};

const std::unordered_map<std::string, onnxEnum> DeviceIDs::string_to_deviceType_ =  {{"NPU", ONNXIFI_DEVICE_TYPE_NPU},
                                                                                     {"DSP", ONNXIFI_DEVICE_TYPE_DSP},
                                                                                     {"GPU", ONNXIFI_DEVICE_TYPE_GPU},
                                                                                     {"CPU", ONNXIFI_DEVICE_TYPE_CPU},
                                                                                     {"FPGA", ONNXIFI_DEVICE_TYPE_FPGA},
                                                                                     {"HETEROGENEOUS", ONNXIFI_DEVICE_TYPE_HETEROGENEOUS}};





struct BackendRep   {
public:
  BackendRep(onnxGraph graph): graph_(graph) {}
 
  //TODO: Implement run
  std::unordered_map<std::string, py::array> run(py::dict inputs, py::kwargs kwargs = {})  {
    return std::unordered_map<std::string, py::array>{};
  }

  ~BackendRep()  {
    if (onnxReleaseGraph(graph_) != ONNXIFI_STATUS_SUCCESS)  {
      throw("Internal error: onnxReleaseGraph failed (expected ONNXIFI_STATUS_SUCCESS)");
    }
  }
private:    
  onnxGraph graph_;  
};

class Backend  {
public:
  Backend() : devices_() {}

  //Parses string and checks if corresponding backendID exists
  bool supports_device(std::string device)  {
    return devices_.getBackendID(device) != NULL;
  }

  std::vector<std::pair<std::string, std::string>> get_device_info() {
    return devices_.getDeviceInfo();
  }

  bool is_compatible(ModelProto model, std::string device="CPU", 
                     py::kwargs kwargs = {})  {
    if (!this->supports_device(device))  {
      return false;
    }
    onnxBackendID id = devices_.getBackendID(device);
    size_t size = model.ByteSizeLong(); 
    char* buffer = new char[size];
    model.SerializeToArray(buffer, size);
    auto compatibilityStatus = onnxGetBackendCompatibility(id, size, buffer);
    delete [] buffer;
    return compatibilityStatus == ONNXIFI_STATUS_SUCCESS || 
           compatibilityStatus == ONNXIFI_STATUS_FALLBACK;
  }

  //TODO: Implement prepare
  //Initializes graph
  //ownership of graph is passed onto BackendRep object
  //Returns BackendRep object
  BackendRep prepare(ModelProto model, std::string device="CPU", py::kwargs kwargs = {})  {
    return BackendRep(NULL);
  } 


  //TODO Implement run_model
  std::unordered_map<std::string, py::array> run_model(ModelProto model, py::dict inputs,
                                                 std::string device = "CPU", py::kwargs kwargs = {})  {
    return std::unordered_map<std::string, py::array>{};
  }

  //TODO: Implement run_node
  std::unordered_map<std::string, py::array> run_node(NodeProto node, py::dict inputs,
                                                 std::string device = "CPU", /*outputs_info = None,*/
                                                 py::kwargs kwargs = {})  {
    return std::unordered_map<std::string, py::array>{};
  }
private:
  DeviceIDs devices_;
};


PYBIND11_MODULE(python_onnxifi, m) {
    py::class_<BackendRep>(m, "BackendRep")
        .def("run", &BackendRep::run);

    py::class_<Backend>(m, "Backend")
        .def(py::init<>())
        .def("is_compatible", &Backend::is_compatible)
        .def("prepare", &Backend::prepare)
        .def("run_model", &Backend::run_model)
        .def("run_node", &Backend::run_node)
        .def("supports_device", &Backend::supports_device)
        .def("get_device_info", &Backend::get_device_info);
}

#include <pybind11/pybind11.h>
#include <string.h>
#include <google/protobuf/message_lite.h>
#include <unordered_map>
#include <utility>
#include <cstdio>
#include "onnx/onnxifi.h"
#include "onnx/onnx_pb.h"

namespace py = pybind11;

struct DeviceIDs  {
public:
  //Get all backendIDs from ONNXIFI interface
  //Fill device_to_onnxBackendID map
  DeviceIDs() :  {
    if (onnxGetBackendIDs(ids_, &num_backends_) != ONNXIFI_STATUS_FALLBACK) {
      throw("Internal error: onnxGetBackendIDs failed (expected ONNXIFI_STATUS_FALLBACK)");
    }
    ids_ = new onnxBackendID[num_backends_];
    if (onnxGetBackendIDs(ids_, &num_backends_) != ONNXIFI_STATUS_SUCCESS) {
      throw("Internal error: onnxGetBackendIDs failed (expected ONNXIFI_STATUS_SUCCESS)");
    }

    for (auto i = 0; i < num_backends_; ++i)  {
      onnxEnum deviceType;
      size_t infoSize = sizeof(onnxEnum);
      if (onnxGetBackendInfo(ids_[i], ONNXIFI_BACKEND_DEVICE_TYPE, 
          &deviceType, &deviceType, &infoSize) != ONNXIFI_STATUS_SUCCESS)  {
        throw("Internal Error: onnxGetBackendInfo failed (expected ONNXIFI_STATUS_SUCCESS)");
      }
      if (deviceType_to_string.find(deviceType) == deviceType_to_string.end())  {
        throw("Internal Error: onnxGetBackendInfo returned an invalid device type");
      }
      device_to_onnxBackendID_[deviceType].push_back(ids_[i]);
    }
  }

  //Returns a vector of pairs - one pair for each backendID
    //First element of the pair is a device handle used from python to run on a device
    //Second element is a device description from the ONNXIFI interface
    //Example of the pair: ("GPU:1", "gpu description")
  std::vector<std::pair<std::string, std::string>> getDeviceInfo()  {
    std::vector<std::pair<std::string, std::string>> deviceInfo;
    for (auto it : device_to_onnxBackendID)  {
      auto& deviceType = it.first;
      auto& deviceList = it.second;
      for (auto i = 0; i < deviceList.size(); ++i)  {
        size_t deviceInfoSize = 0;
        if (onnxGetBackendInfo(deviceList[i], ONNXIFI_BACKEND_DEVICE,
                               NULL, &deviceInfoSize) != ONNXIFI_STATUS_FALLBACK)  {
          throw("Internal Error: onnxGetBackendInfo failed (expected ONNXIFI_STATUS_FALLBACK)");
        }
        char* deviceInfo = new char[deviceInfoSize];
        if (onnxGetBackendInfo(deviceList[i], ONNXIFI_BACKEND_DEVICE,
                               deviceInfo, &deviceInfoSize) != ONNXIFI_STATUS_SUCCESS)  {
          throw("Internal Error: onnxGetBackendInfo failed (expected ONNXIFI_STATUS_SUCCESS)");
        }
        std::string deviceHandle = deviceType_to_string[deviceType] + ":" + std::to_string(i + 1);
        deviceInfo.push_back(std::pair(deviceHandle, deviceInfo));
      }
    }
    return deviceInfo;
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
    if (string_to_deviceType.find(deviceString) == string_to_deviceType.end())  {
      return NULL;
    }
    auto deviceType = string_to_deviceType[deviceString];
    //Treating ID of 0 as 1 and bringing 1-indexing to 0-indexing
    if (deviceID > 0) {
      deviceID--;
    }
    if (deviceID < 0 || deviceID >= device_to_onnxBackendID[deviceType].size())  {
      return NULL;
    }
    return device_to_onnxBackendID[deviceType][deviceID];
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
  ~BackendIDs()  {
    for (auto backend : initialized_) {
      if (onnxReleaseBackend(backend.second) != ONNXIFI_STATUS_SUCCESS)  {
        throw("Internal error: onnxReleaseBackend failed (expected ONNXIFI_STATUS_SUCCESS)");
      }
    }
    if (!ids_) {
      for (auto i = 0; i < num_backends_; ++i)  {
        if (onnxReleaseBackendIDs(ids_[i]) != ONNXIFI_STATUS_SUCCESS)  {
          throw("Internal error: onnxReleaseBackendIDs failed (expected ONNXIFI_STATUS_SUCCESS)");
        }
      }
      delete [] ids_;
    }
  }

  //Convenience maps to map betweeen onnxEnum and string
  //  python user will input the string
  //  C++ internals and ONNXIFI represent as onnxEnum
  static const std::unordered_map<onnxEnum, std::string> deviceType_to_string = 
                                                {{ONNXIFI_DEVICE_TYPE_NPU, "NPU"},
                                                 {ONNXIFI_DEVICE_TYPE_DSP, "DSP"},
                                                 {ONNXIFI_DEVICE_TYPE_GPU, "GPU"},
                                                 {ONNXIFI_DEVICE_TYPE_CPU, "CPU"},
                                                 {ONNXIFI_DEVICE_TYPE_FPGA, "FPGA"},
                                                 {ONNXIFI_DEVICE_TYPE_HETEROGENEOUS, "HETEROGENEOUS"}};

  static const std::unordered_map<onnxEnum, std::string> string_to_deviceType = 
                                                {{"NPU", ONNXIFI_DEVICE_TYPE_NPU},
                                                 {"DSP", ONNXIFI_DEVICE_TYPE_DSP},
                                                 {"GPU", ONNXIFI_DEVICE_TYPE_GPU},
                                                 {"CPU", ONNXIFI_DEVICE_TYPE_CPU},
                                                 {"FPGA", ONNXIFI_DEVICE_TYPE_FPGA},
                                                 {"HETEROGENEOUS", ONNXIFI_DEVICE_TYPE_HETEROGENEOUS}};
 
private: 
  //Array of all backendIDs
  onnxBackendID* ids_;
  size_t num_backends_;

  //device_to_onnxBackendID[deviceType][deviceID - 1] will give corresponding onnxBackendID
  //1-indexed to conform to onnx/onnx/backend/base.py
  //deviceID of 0 refers to any onnxBackendID of that deviceType
  std::unordered_map<onnxEnum, vector<onnxBackendID>> device_to_onnxBackendID_;

  //If the backendID has been initialized, will map to corresponding backendID pointer
  //Otherwise, not present
  std::unordered_map<onnxBackendID, onnxBackend> initialized_;
}









struct BackendRep()   {
public:
  BackendRep(onnxGraph&& graph): graph_(graph) {}
  std::vector<onnxTensorDescriptor> run(std::vector<onnxTensorDescriptor> inputs, py::kwargs)  {

  }

  ~BackendRep()  {
    if (onnxReleaseGraph(graph_) != ONNXIFI_STATUS_SUCCESS)  {
      throw("Internal error: onnxReleaseGraph failed (expected ONNXIFI_STATUS_SUCCESS)");
    }
  }
private:    
  onnxGraph graph_;  
}

class Backend  {
public:
  static DeviceIDs devices_;

  bool is_compatible(ModelProto model, char* device="CPU", 
                     py::kwargs)  {
    if (!this->supports_device(device))  {
      return false;
    }
    size_t size = model.ByteSizeLong(); 
    char* buffer = new char[size];
    model.SerializeToArray(buffer, size);
    auto compatibilityStatus = onnxGetBackendCompatibility(backendID, size, buffer);
    delete [] buffer;
    return compatibilityStatus == ONNXIFI_STATUS_SUCCESS || 
           compatibilityStatus == ONNXIFI_STATUS_FALLBACK;
  }

  //TODO: Weight Descriptors
  //Initializes graph
  //ownership of graph is passed onto BackendRep object
  //Returns pointer to BackendRep object - ownership must be passed to python
  //so GC cleans up
  BackendRep* prepare(ModelProto model, device="CPU", py::kwargs)  {
    onnxBackend backend = devices_->prep_device(device);
    if (!backend)  {
      return nullptr; 
    }
    std::string data;
    model.SerializeToString(&data);
    const void* serializedModel = data.c_str();
    size_t serializedModelSize = data.size();
    onnxGraph graph;
    if (onnxInitGraph(backend, serializedModelSize, serializedModel, 0, nullptr, &graph)
                                                              != ONNXIFI_STATUS_SUCCESS)  {
      throw("Internal error: onnxInitGraph failed (expected ONNXIFI_STATUS_SUCCESS)");
    }

    return new BackendRep(std::move(graph));
  } 


    def run_model(cls,
                  model,  # type: ModelProto
                  inputs,  # type: Any
                  device='CPU',  # type: Text
                  **kwargs  # type: Any
                  )


    def run_node(cls,
                 node,  # type: NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Dict[Text, Any]
                 )

  //Parses string and checks if corresponding backendID exists
  bool supports_device(char* device)  {
    return devices_.getBackendID(device) != NULL;
  }

private:
}

Backend::DeviceIDs devices_ = DeviceIDs();

PYBIND11_MODULE(example, m) {
    py::class_<BackendIDs>(m, "BackendIDs")
        .def(py::init<>())
        .def("getDeviceType", &BackendIDs::getDeviceType);
}

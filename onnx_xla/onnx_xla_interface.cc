#include "onnx/onnxifi.h"
#include "onnx_xla/backend.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

//TODO: Robust error handling
//TODO: Figure out how to determine type of device, what information to store
//      about hardware, and how to modify execution as a result

onnxStatus onnxifiTryCatch(std::function<onnxStatus()> tryBlock)  {
  try  {                                                   
    return tryBlock();          
  }
  catch (const std::bad_alloc& e) {                                            
    std::cout << "Allocation failed: " << e.what() << std::endl;               
    return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;                                    
  }                                                                            
  catch (const std::exception &e) {                                            
    std::cerr << "Internal Error: " << e.what() << std::endl;                  
    return ONNXIFI_STATUS_INTERNAL_ERROR;                                      
  }                                                                            
  catch (...) {                                                                
    return ONNXIFI_STATUS_INTERNAL_ERROR;                                      
  }
}

//TODO: More formal representation of backendID - CPU, GPU, TPU?
struct OnnxXlaBackendID {
  int device_id{0};
};

struct EventControl {
  EventControl() : signalled_(false) {}
  volatile bool signalled_;
  std::mutex mutex_;
  std::condition_variable condvar_;
};


//Backend engine
//  backendID will eventually determine translation detail
struct BackendControl {
public:
  BackendControl(OnnxXlaBackendID* id) : backendID(id) {}
  //use OnnxParser and XlaTransform to return executor
  onnxStatus build(const void* serializedModel, size_t serializedModelSize, uint32_t weightsCount, 
                   const onnxTensorDescriptor *weightDescriptors, onnxGraph* graph) {
   
    onnx_xla::OnnxParser parser(serializedModel, serializedModelSize);
    std::unique_ptr<ONNX_NAMESPACE::Graph> ir(nullptr);
    auto parseStatus = parser.parse(ir);
    if (parseStatus != ONNXIFI_STATUS_SUCCESS)  {
      return parseStatus;
    }
    std::string build_name = ir->name();
    onnx_xla::XlaTransform runner(std::move(ir), build_name,
                                  weightsCount, weightDescriptors);
    auto translateStatus = runner.translateGraph();
    if (translateStatus != ONNXIFI_STATUS_SUCCESS)  {
      return translateStatus;
    }

    *graph = reinterpret_cast<onnxGraph>(runner.executor());
    return ONNXIFI_STATUS_SUCCESS;
  }
private:
  OnnxXlaBackendID* backendID;
};

//Create 1 backendID
//TODO: Determining # of CPU, GPU, TPU devices and return
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendIDs)(onnxBackendID *backendIDs, size_t *numBackends) {
  onnxifiTryCatch([&] {
    *numBackends = 0;
    *backendIDs = reinterpret_cast<onnxBackendID>(new OnnxXlaBackendID());
    *numBackends = 1;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

//Free memory for given backend ID
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseBackendID)(onnxBackendID backendID) {
  onnxifiTryCatch([&] {
    if (!backendID) {
      return ONNXIFI_STATUS_INVALID_ID;
    }
    auto *backend_id = reinterpret_cast<OnnxXlaBackendID *>(backendID);
    delete backend_id;
    return ONNXIFI_STATUS_SUCCESS;
  });
}


//Returning info for given ID
//TODO: Make sure this information is correct
//TODO: Expand for different IDs (TPU/GPU) in the future
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendInfo)(onnxBackendID backendID, onnxBackendInfo infoType,
                        void *infoValue, size_t *infoValueSize) {
  onnxifiTryCatch([&] {
    if (!infoValueSize) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }

    if (!backendID)  {
      return ONNXIFI_STATUS_INVALID_ID;
    }

    auto SET_STRING = [&](const char* str)  {
                        if (!infoValue || *infoValueSize < strlen(str) + 1)  {
                          *infoValueSize = strlen(str) + 1;
                          return ONNXIFI_STATUS_FALLBACK;
                        }
                        strncpy((char *)(infoValue), str, *infoValueSize);
                        *infoValueSize = strlen(str) + 1;
                        return ONNXIFI_STATUS_SUCCESS;
                      };
    auto SET_UINT64 = [&](uint64_t x)  {
                        if (!infoValue || *infoValueSize < sizeof(uint64_t))  {
                          *infoValueSize = sizeof(uint64_t);
                          return ONNXIFI_STATUS_FALLBACK;
                        }
                        *(uint64_t *)(infoValue) = x;                                              
                        *infoValueSize = sizeof(uint64_t);
                        return ONNXIFI_STATUS_SUCCESS;
                      };

    switch(infoType)  {
      case ONNXIFI_BACKEND_NAME:  {
        return SET_STRING("onnx-xla");
      }
      case ONNXIFI_BACKEND_VENDOR:  {
        return SET_STRING("Google");
      }
      case ONNXIFI_BACKEND_VERSION:  {
        return SET_STRING("1.0.0");
      }
      case ONNXIFI_BACKEND_EXTENSIONS:  {
        *infoValueSize = 0;
        return ONNXIFI_STATUS_SUCCESS;
      }
      case ONNXIFI_BACKEND_DEVICE:  {
        return SET_STRING("cpu (for now in development)");
      }
      case ONNXIFI_BACKEND_DEVICE_TYPE:  {
        return SET_UINT64(ONNXIFI_DEVICE_TYPE_CPU);
      }
      case ONNXIFI_BACKEND_CAPABILITIES:  {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_INIT_PROPERTIES:  {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_MEMORY_TYPES:  {
        return SET_UINT64(ONNXIFI_MEMORY_TYPE_CPU);
      }
      case ONNXIFI_BACKEND_MEMORY_SIZE:  {
        //TODO
        return ONNXIFI_STATUS_UNSUPPORTED_PARAMETER;
      }
      case ONNXIFI_BACKEND_MAX_GRAPH_SIZE: {
        return SET_UINT64(1000000UL);
      }
      case ONNXIFI_BACKEND_MAX_GRAPH_COUNT: {
        return SET_UINT64(1UL);
      }
      case ONNXIFI_BACKEND_MACS_FP32: {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_MACS_FP16:  {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_MEMORY_BANDWIDTH:  {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH:  {
        return SET_UINT64(0UL);
      }
      default:  {
        return ONNXIFI_STATUS_UNSUPPORTED_PARAMETER;
      }
    }
  });
}

//TODO: Figure out how to get compatibility e.g. sufficient to run OnnxParser?
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendCompatibility)(onnxBackendID backendID, size_t onnxModelSize,
                                 const void *onnxModel) {
  onnxifiTryCatch([&] {
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }
    return ONNXIFI_STATUS_SUCCESS;
  });
}

//TODO: any arguments to pass?
//Create and return a BackendControl object
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitBackend)(onnxBackendID backendID, const uint64_t *auxPropertiesList,
                     onnxBackend *backend) {
  onnxifiTryCatch([&] {
    auto *backend_id = reinterpret_cast<OnnxXlaBackendID *>(backendID);
    *backend = reinterpret_cast<onnxBackend>(new BackendControl(backend_id));
    return ONNXIFI_STATUS_SUCCESS;
  });
}

//Release BackendControl object
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxReleaseBackend)(onnxBackend backend) {
  onnxifiTryCatch([&] {    
    if (!backend) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    auto *backendController = reinterpret_cast<BackendControl *>(backend);
    delete backendController;
    return ONNXIFI_STATUS_SUCCESS;
  });
}


//Create and return XlaExecutor object
// TODO: Ignore the weightDescriptors for now and rely on initialization list
//TODO: error handling with error status codes passed
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitGraph)(onnxBackend backend, size_t onnxModelSize,
                   const void *onnxModel, uint32_t weightsCount,
                   const onnxTensorDescriptor *weightDescriptors,
                   onnxGraph *graph) {
  onnxifiTryCatch([&] {
    *graph = NULL;
    if (!backend) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }
    auto *backendController = reinterpret_cast<BackendControl *>(backend);
    return backendController->build(onnxModel, onnxModelSize, weightsCount, 
                                    weightDescriptors, graph);
  });
}

//Verify IO metadata and use initIO to store location of IO
//TODO: memoryType field ignored for now
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxSetGraphIO)(onnxGraph graph, uint32_t inputsCount,
                    const onnxTensorDescriptor *inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptor *outputDescriptors) {
  onnxifiTryCatch([&] {
    if (!graph) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    if (!inputDescriptors || !outputDescriptors) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    auto *executor = reinterpret_cast<onnx_xla::XlaExecutor*>(graph);
    return executor->initIO(inputsCount, inputDescriptors, outputsCount, outputDescriptors);
  });
}

//Runs the XlaExecutor by sending literals to server and executing computation 
//TODO: support for synchronization primitives; For now assume, they are always set
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxRunGraph)(onnxGraph graph, const onnxMemoryFence *inputFence,
                  onnxMemoryFence *outputFence) {
 onnxifiTryCatch([&] {
    if (!graph) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    auto *executor = reinterpret_cast<onnx_xla::XlaExecutor *>(graph);
    auto sendInputsStatus = executor->sendInputs();
    if (sendInputsStatus != ONNXIFI_STATUS_SUCCESS)  {
      return sendInputsStatus;
    }
    return executor->executeComputation();
  });
}

//Frees executor memory
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxReleaseGraph)(onnxGraph graph) {
  onnxifiTryCatch([&] {
    if (!graph) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    auto *executor = reinterpret_cast<onnx_xla::XlaExecutor *>(graph);
    delete executor;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

//Initialize event by creating EventControl object on the heap
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitEvent)(onnxBackend backend, onnxEvent* event) {
  onnxifiTryCatch([&] {
    if (!event)  {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    *event = NULL;

    if (!backend)  {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }

    *event = reinterpret_cast<onnxEvent>(new EventControl());
    return ONNXIFI_STATUS_SUCCESS;
  });
}

//Signal Event by changing the signalled boolean under mutex hold
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxSignalEvent)(onnxEvent event)  {
  onnxifiTryCatch([&] {    
    if (!event)  {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    auto *eventController = reinterpret_cast<EventControl *>(event);
    {
       std::lock_guard<std::mutex> lk(eventController->mutex_);
       if (eventController->signalled_)  {
         return ONNXIFI_STATUS_INVALID_STATE;
       }
       eventController->signalled_ = true;
    }
    eventController->condvar_.notify_all();
    return ONNXIFI_STATUS_SUCCESS;
  });
}

//Wait for signalled to be turned true using conditional variable to coordinate
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxWaitEvent)(onnxEvent event)  {
  onnxifiTryCatch([&] {
    if (!event)  {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    
    auto *eventController = reinterpret_cast<EventControl *>(event);  
    std::unique_lock<std::mutex> lk(eventController->mutex_);
    eventController->condvar_.wait(lk, [&eventController]
                                       {return eventController->signalled_;});
        
    return ONNXIFI_STATUS_SUCCESS;
  });
}

//Free memory that was allocated for the EventControl object
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxReleaseEvent)(onnxEvent event)  {
  onnxifiTryCatch([&] {
    if (!event)  {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    auto *eventController = reinterpret_cast<EventControl *>(event);
    delete eventController;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

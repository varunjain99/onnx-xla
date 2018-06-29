#include "onnx/onnxifi.h"
#include "onnx_xla/backend.h"

struct OnnxXlaBackendID {
  int device_id{0};
};

struct BackendControl {
public:
  onnxStatus build(void const *serialized_onnx_model,
                         size_t serialized_onnx_model_size) {
    return ONNXIFI_STATUS_SUCCESS;
  }

  onnx_xla::XlaExecutor* executor() {
    return transformer_.executor();
  }

private:
  onnx_xla::XlaTransform* transformer_{nullptr};
  onnx_xla::OnnxParser* parser_{nullptr};
};

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendIDs)(onnxBackendID *backendIDs, size_t *numBackends) {
  try {
    *backendIDs = (onnxBackendID)(new OnnxXlaBackendID());
    *numBackends = 1;
    return ONNXIFI_STATUS_SUCCESS;
  }
  ONNXIFI_CATCH_EXCPETION();
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseBackendID)(onnxBackendID backendID) {
  try {
    auto *backend_id = reinterpret_cast<OnnxXlaBackendID *>(backendID);
    if (!backend_id) {
      return ONNXIFI_STATUS_INVALID_ID;
    }
    delete backend_id;
    return ONNXIFI_STATUS_SUCCESS;
  }
  ONNXIFI_CATCH_EXCPETION();
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendInfo)(onnxBackendID backendID, onnxBackendInfo infoType,
                        void *infoValue, size_t *infoValueSize) {
  try {
    if (!infoValueSize) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
#define SET_STRING(str)                                                        \
  {                                                                            \
    snprintf((char *)(infoValue), *infoValueSize, str);                        \
    *infoValueSize = strlen(str) + 1;                                          \
  }

#define SET_UINT64(x)                                                          \
  {                                                                            \
    if (*infoValueSize < sizeof(uint64_t)) {                                   \
      return ONNXIFI_STATUS_INVALID_POINTER;                                   \
    }                                                                          \
    *(uint64_t *)(infoValue) = x;                                              \
    *infoValueSize = sizeof(uint64_t);                                         \
  }
    if (infoType == ONNXIFI_BACKEND_NAME) {
      SET_STRING("OnnxXla");
    } else if (infoType == ONNXIFI_BACKEND_VENDOR) {
      SET_STRING("Google");
    } else if (infoType == ONNXIFI_BACKEND_VERSION) {
      SET_STRING("1.0.0");
    } else if (infoType == ONNXIFI_BACKEND_EXTENSIONS) {
      *infoValueSize = 0;
    } else if (infoType == ONNXIFI_BACKEND_DEVICE) {
      SET_STRING("cpu (for now in development)");
    } else if (infoType == ONNXIFI_BACKEND_DEVICE_TYPE) {
      SET_UINT64(ONNXIFI_DEVICE_TYPE_CPU);
    } else if (infoType == ONNXIFI_BACKEND_CAPABILITIES) {
      //SET_UINT64(0UL);
    } else if (infoType == ONNXIFI_BACKEND_INIT_PROPERTIES) {
      //SET_UINT64(0UL);
    } else if (infoType == ONNXIFI_BACKEND_MEMORY_TYPES) {
      SET_UINT64(ONNXIFI_MEMORY_TYPE_CPU);
    } else if (infoType == ONNXIFI_BACKEND_MEMORY_SIZE) {
      /*size_t free, total;
      if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
      }
      SET_UINT64(uint64_t(total));*/
    }
    // Dummy numbers
    else if (infoType == ONNXIFI_BACKEND_MAX_GRAPH_SIZE) {
      //SET_UINT64(1000000UL);
    } else if (infoType == ONNXIFI_BACKEND_MAX_GRAPH_COUNT) {
      //SET_UINT64(1UL);
    } else if (infoType == ONNXIFI_BACKEND_MACS_FP32) {
      //SET_UINT64(0UL);
    } else if (infoType == ONNXIFI_BACKEND_MACS_FP16) {
      //SET_UINT64(0UL);
    } else if (infoType == ONNXIFI_BACKEND_MEMORY_BANDWIDTH) {
      //SET_UINT64(0UL);
    } else if (infoType == ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH) {
      //SET_UINT64(0UL);
    } else if (infoType == ONNXIFI_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH) {
      //SET_UINT64(0UL);
    } else {
      return ONNXIFI_STATUS_UNSUPPORTED_PARAMETER;
    }
    return ONNXIFI_STATUS_SUCCESS;
#undef RETURN_STRING
#undef SET_UINT64
  }

  ONNXIFI_CATCH_EXCPETION();
}

// NB: Why not have onnxModel as const char*?  and we should set the name to
// onnxGraph And we don't have ir_version and opset info here, which are needed
// for model check
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendCompatibility)(onnxBackendID backendID, size_t onnxModelSize,
                                 const void *onnxModel) {
  try {
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }

    return ONNIXIFI_STATUS_SUCCESS; //later we should implement this error analysis

    // NB: not ideal case. We CHECK model by actually trying to run the
    // conversion. However, this might be the case for other vendors
    //OnnxTensorRTBackendRep backendrep;
    //return backendrep.ImportModel(onnxModel, onnxModelSize);
  }
  ONNXIFI_CATCH_EXCPETION();
}

// NB: Passing arguments to backend is tricky. And we need more documentation
// for it I didn't put any arguments here for now.
// TODO: submit arguments for
// - setMaxBatchSize (size_t)
// - setMaxWorkspaceSize (size_t)
// - setHalf2Mode (bool)
// - setInt8Mode (bool)
// - setDebugSync (bool)
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitBackend)(onnxBackendID backendID, const uint64_t *auxPropertiesList,
                     onnxBackend *backend) {
  try {
    *backend = (onnxBackend)(new BackendControl());
    return ONNXIFI_STATUS_SUCCESS;
  }
  ONNXIFI_CATCH_EXCPETION();
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseBackend)(onnxBackend backend) {
  try {
    auto *backendcontroller = reinterpret_cast<BackendControl *>(backend);
    if (!backendcontroller) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    delete backendcontroller;
    return ONNXIFI_STATUS_SUCCESS;
  }
  ONNXIFI_CATCH_EXCPETION();
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitGraph)(onnxBackend backend, size_t onnxModelSize,
                   const void *onnxModel, uint32_t weightsCount,
                   const onnxTensorDescriptor *weightDescriptors,
                   onnxGraph *graph) {
  try {
    auto *backendcontroller = reinterpret_cast<BackendControl *>(backend);
    if (!backendcontroller) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }

    // Parse the model
    // TODO: Ignore the weightDescriptors for now and rely on initialization list
    // this will take ModelProto -> IR Graph -> XlaComputation
    auto ret = backendcontroller->build(onnxModel, onnxModelSize);
    if (ret != ONNXIFI_STATUS_SUCCESS) {
      return ret;
    }

    // return XlaExecutor which has the XlaComputation and literals to send to
    // the server
    //TODO: error handling
    *graph = (*onnxGraph)(backendcontroller->executor());
    return ONNXIFI_STATUS_SUCCESS;
  }
  ONNXIFI_CATCH_EXCPETION();
}


ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxSetGraphIO)(onnxGraph graph, uint32_t inputsCount,
                    const onnxTensorDescriptor *inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptor *outputDescriptors) {
  try {
    auto *exector = reinterpret_cast<onnx_xla::XlaExecutor*>(graph);
    if (!executor) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    if (!inputDescriptors || !outputDescriptors) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }

    return executor->initIO(inputsCount, inputDescriptors, outputsCount,
                             outputDescriptors);
  }
  ONNXIFI_CATCH_EXCPETION();
}

//For now assume, synchronization primitives are always set
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxRunGraph)(onnxGraph graph, const onnxMemoryFence *inputFence,
                  onnxMemoryFence *outputFence) {
  try {
    auto *executor = reinterpret_cast<XlaExecutor *>(graph);
    if (!executor) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }

    executor->sendLiterals();
    executor->executeComputation();
    return ONNXIFI_STATUS_SUCCESS;
  }
  ONNXIFI_CATCH_EXCPETION();
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseGraph)(onnxGraph graph) {
  try {
    auto *executor = reinterpret_cast<XlaExecutor *>(graph);
    if (!executor) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    delete exeuctor;
    return ONNXIFI_STATUS_SUCCESS;
  }
  ONNXIFI_CATCH_EXCPETION();
}

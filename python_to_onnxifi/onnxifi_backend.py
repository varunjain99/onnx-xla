from build.python_onnxifi import *
import onnx
from onnx import (NodeProto,
                  ModelProto)

class OnnxifiBackendRep(object):
    #Constructor never be used - underlying C++ object created behind the scenes
    #in OnnxifiBacken::prepare
    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        self.backend_rep_.run(inputs, **kwargs)


class OnnxifiBackend(object):
    def __init__(self):
        self.backend_ = Backend()

    def is_compatible(self,
                      model,  # type: ModelProto
                      device='CPU',  # type: Text
                      **kwargs  # type: Any
                      ):  # type: (...) -> bool
        self.backend_(model.SerializeToString(), device, **kwargs)

    def prepare(self,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> BackendRep
        onnx.checker.check_model(model)
        return self.backend_(model.SerializeToString(), device, **kwargs)

    def run_model(self,
                  model,  # type: ModelProto
                  inputs,  # type: Any
                  device='CPU',  # type: Text
                  **kwargs  # type: Any
                  ):  # type: (...) -> Tuple[Any, ...]
        backendRep = self.prepare(model, device, **kwargs)
        assert backendRep is not None
        return backendRep.run(inputs, **kwargs)

    #TODO: Implement run_node
    def run_node(cls,
                 node,  # type: NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Dict[Text, Any]
                 ):  # type: (...) -> Optional[Tuple[Any, ...]]
        return None

    def supports_device(self, device):  # type: (Text) -> bool
        self.backend_.supports_device(device)

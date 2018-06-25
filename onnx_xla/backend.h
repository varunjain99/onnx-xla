#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "onnx/onnx/common/ir.h"

using ONNX_NAMESPACE;
using xla;

namespace onnx-xla {
  class XlaTransform final  {
  public:
    XlaTransform(const Graph& ir_, const string name) :
      ir_(ir), builder_(XlaBuilder(name)) {}

    ~XlaTransform() {
      for (const auto& l : literals_) {
        delete l;
      }
    }
  private:
    const Graph& ir_;
    XlaBuilder builder_;
    vector<Literal*> literals_;
    std::unordered_map<const Value*, XlaOp*> value_to_op_;

    //returns pointer to Literal, NULL if error
    Literal* initializerToLiteral(const Tensor& t);
    //true if successful, false if not
    bool intializersToLiterals();
    void computeBuild();
  }
}

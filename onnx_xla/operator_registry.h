#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/onnxifi.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"

#include "onnx_xla/types.h"

#include <functional>
#include <utility>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace onnx_xla {

using ::xla::Literal;
using ::xla::ShapeUtil;
using ::xla::Shape;
using ::xla::primitive_util::NativeToPrimitiveType;
using ::xla::XlaOp;
using ::xla::XlaBuilder;
using ::xla::XlaComputation;
using ::xla::LiteralBase;
using ::xla::StatusOr;
using ::xla::Padding;
using ::xla::PrimitiveType;

using ::ONNX_NAMESPACE::Value;
using ::ONNX_NAMESPACE::Dimension;
using ::ONNX_NAMESPACE::Symbol;
using ::ONNX_NAMESPACE::Node;

using ValueOpMap = std::unordered_map<const Value*, XlaOp>;
using TranslationFunction =
    std::function<onnxStatus(const Node&, XlaBuilder&, ValueOpMap&)>;
using TranslationMap = std::unordered_map<Symbol, TranslationFunction>;

// Class for registry of ONNX operators with corresponding translation functions
class OperatorRegistry final {
 public:
  // Use constructor (through macro) to register translator at static time
  class OperatorRegisterOnce final {
   public:
    OperatorRegisterOnce(const Symbol& nodeKind,
                         TranslationFunction translator);
  };

  // Translate given node
  // Updates builder
  // In: Expect valueToOp to exist for every node input
  // Out: Expect valueToOp to be assigned for every node output
  onnxStatus translate(const Node& n,
                       XlaBuilder& builder,
                       ValueOpMap& valueToOp);
  // Returns reference to static singleton registry
  static OperatorRegistry& registry();

  // Utilities to help with operator translation
  static XlaComputation max(PrimitiveType dataType);
  static XlaComputation add(PrimitiveType dataType);

  // Converts input sizes vector of Dimension into vector of int64
  // Throws if not possible (missing shape dimensions)
  static std::vector<int64> parseOnnxInputSizes(const Node& n,
                                                size_t inputIndex);

  // Given two XlaOps, returns a broadcast dimensions vector required by the
  // XlaBuilder to perform elementary binary operations that support
  // multidirectional broadcasting
  static std::vector<int64> getMultidirectionalBroadcastArg(
      const XlaBuilder& builder,
      const XlaOp& firstOp,
      const XlaOp& secondOp);

 private:
  // Singleton instance should only be made in the class
  OperatorRegistry() = default;
  // Wrapper for registry map - should not be directly accessed
  // Register in map through below macro
  // Execute translation function through registry()->executeTranslation
  static TranslationMap& map();
};

// Use this macro to register Symbol("name") with translator of type
// TranslationFunction
#define REGISTER_OPERATOR_TRANSLATOR(name, translator)                        \
  static OperatorRegistry::OperatorRegisterOnce register##name(Symbol(#name), \
                                                               translator);
}

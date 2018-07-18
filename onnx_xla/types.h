#pragma once

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "onnx/onnx_pb.h"
#include <complex>
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"

#include <Eigen/Core>
#include "onnx/onnxifi.h"
#include <unordered_map>

namespace onnx_xla {
// Get access to XLA types used
using ::tensorflow::string;

using ::tensorflow::int8;
using ::tensorflow::int16;
using ::tensorflow::int32;
using ::tensorflow::int64;

using ::tensorflow::bfloat16;

using ::tensorflow::uint8;
using ::tensorflow::uint16;
using ::tensorflow::uint32;
using ::tensorflow::uint64;

using complex64 = std::complex<float>;

using ::Eigen::half;

// Helper functions to translate between ONNX and XLA types
xla::PrimitiveType onnxToPrimitive(
    const ONNX_NAMESPACE::TensorProto_DataType& data_type);
}

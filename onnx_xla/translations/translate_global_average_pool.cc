#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Compute GlobalAveragePool
// 1. Use Reduce window with add computation
// 2. Divide by size of window 
onnxStatus translateGlobalAveragePool(const Node& n,
                         XlaBuilder& builder,
                         ValueOpMap& valueToOp) {

   auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());
   //TODO: Use OperatoryRegistry add in Softmax PR
   XlaComputation add;
   {
     XlaBuilder builder("add");
     auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
     auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
     builder.Add(y, x);
     add = builder.Build().ConsumeValueOrDie();
   }

   //TODO: Use static function from Pooling PR to set window
   std::vector<int64> windowDimensions =

   std::vector<int64> windowStrides(windowDimensions.size(), 1);
   
  PoolOp = builder.ReduceWindow(valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::Zero(dataType)), add,
      windowDimensions, windowStrides, Padding::kValid);


  //TODO: Use Gemm PR to complete this
  auto numWindowElements = std::accumulate(windowDimensions.begin(), windowDimensions.end(), 1L, std::multiplies<int64>());
  //TODO: Make numWindowElements an XlaOp
  valueToOp[n.outputs().at(0)] = builder.Div(PoolOp, ____);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(GlobalAveragePool, translateGlobalAveragePool)
}


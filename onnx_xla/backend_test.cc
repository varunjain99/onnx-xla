#include "onnx_xla/backend.h"
#include "onnx_xla/backend_test.h"
#include <stdlib.h>
#include <cmath>

namespace onnx_xla  {

  bool almost_equal(double a, double b, double epsilon) {
    return std::abs(a - b) < epsilon;
  }

  bool almost_equal(float a, float b, float epsilon)  {
    return std::abs(a - b) < epsilon;
  }
 
  void relu_test()  {
    //Set up IR graph
    Graph relu_graph;
    relu_graph.setName("relu_graph");
    Tensor initializer;
    initializer.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    std::uniform_real_distribution<double> unif(-0.5, 0.5);
    for (int i = 0; i < 24; ++i)  {
      std::random_device rand_dev;          
      std::mt19937 rand_engine(rand_dev()); 
      initializer.doubles().push_back(unif(rand_engine));          
    }
    initializer.sizes().push_back(2);
    initializer.sizes().push_back(3);
    initializer.sizes().push_back(4);
    relu_graph.addInitializerAndInput(initializer, "relu_input");   
    auto relu_node = relu_graph.create(Symbol("Relu"), relu_graph.inputs());
    relu_graph.appendNode(relu_node);     
    relu_graph.return_node()->addInput(relu_node->output());

    //Execute using XLA backend
    XlaTransform runner(relu_graph, "relu");
    runner.initializersToLiterals();
    runner.sendLiterals();
    runner.translateGraph();
    auto results = runner.executeComputation();
    
    //Check correctness
    for (int i = 0; i < 24; ++i)  {
      if (initializer.doubles()[i] > 0.0)  {
        ONNX_ASSERT(almost_equal(initializer.doubles()[i], results[0].data<double>()[i]));  
      } else {
        ONNX_ASSERT(almost_equal(0.0, results[0].data<double>()[i]));
      }
    }   
  }
}

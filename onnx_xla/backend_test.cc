#include "onnx_xla/backend.h"
#include "onnx_xla/backend_test.h"
#include <stdlib.h>
#include <cmath>

namespace onnx_xla  {

  bool almost_equal(float a, float b, float epsilon)  {
    return std::abs(a - b) < epsilon;
  }

  void static_relu_test()  {
    //Set up IR graph
    Graph relu_graph;
    relu_graph.setName("relu_graph");
    Tensor initializer;
    initializer.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    std::uniform_real_distribution<float> unif(-0.5, 0.5);
    for (int i = 0; i < 24; ++i)  {
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      initializer.floats().push_back(unif(rand_engine));
    }
    initializer.sizes().push_back(2);
    initializer.sizes().push_back(3);
    initializer.sizes().push_back(4);
    relu_graph.addInitializerAndInput(initializer, "relu_input");
    auto relu_node = relu_graph.create(Symbol("Relu"), relu_graph.inputs());
    relu_graph.appendNode(relu_node);
    relu_graph.return_node()->addInput(relu_node->output());

    //Set up IO information
    uint64_t* shape = new uint64_t[3];
    shape[0] = 2;
    shape[1] = 3;
    shape[2] = 4;
    onnxTensorDescriptor output;
    output.name = new char[5];
    output.name = "relu";
    output.dataType = ONNXIFI_DATATYPE_FLOAT32;
    output.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    output.dimensions = 3;
    output.shape = shape;
    output.buffer = (onnxPointer) new float[24];

    //Execute using XLA backend
    XlaTransform runner(relu_graph, "relu");
    runner.translateGraph();
    auto executor = runner.executor();
    exector->initIO(0, NULL, 1, output);
    delete [] output.name;
    delete [] shape;
    executor->sendLiterals();
    executor->executeComputation();

    float* answer = (float*) output.buffer;
    //Check correctness
    for (int i = 0; i < 24; ++i)  {
      if (initializer.doubles()[i] > 0.0f)  {
        ONNX_ASSERT(almost_equal(initializer.floats()[i], answer[i]));
      } else {
        ONNX_ASSERT(almost_equal(0.0f, answer[i]));
      }
    }
    delete [] output.buffer;
  }

  void dynamic_relu_test()  {
    //Set up IR graph
    Graph relu_graph;
    relu_graph.setName("relu_graph");

    Value* relu_input = relu_graph.addInput();
    relu_input->setElemType(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
    vector<Dimension> sizes;
    sizes().push_back(2);
    sizes().push_back(3);
    sizes().push_back(4);
    relu_input->setSizes(sizes);
    relu_input->setUniqueName("relu_input");
    auto relu_node = relu_graph.create(Symbol("Relu"), relu_graph.inputs());
    relu_graph.appendNode(relu_node);
    relu_graph.return_node()->addInput(relu_node->output());

    //Set up IO information
    uint64_t* shape = new uint64_t[3];
    shape[0] = 2;
    shape[1] = 3;
    shape[2] = 4;
    onnxTensorDescriptor output;
    output.name = new char[12];
    output.name = "relu_output";
    output.dataType = ONNXIFI_DATATYPE_FLOAT32;
    output.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    output.dimensions = 3;
    output.shape = shape;
    output.buffer = (onnxPointer) new float[24];
    input.name = new char[12];
    output.name = "relu_input";
    input.dataType = ONNXIFI_DATATYPE_FLOAT32;
    input.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    input.dimensions = 3;
    input.shape = shape;
    input.buffer = (onnxPointer) new float[24];

    //Execute using XLA backend
    XlaTransform runner(relu_graph, "relu");
    runner.translateGraph();
    auto executor = runner.executor();
    exector->initIO(0, NULL, 1, output);
    delete [] output.name;
    delete [] shape;
    float* input_ptr = (float*) input.buffer;
    std::uniform_real_distribution<float> unif(-0.5, 0.5);
    for (int i = 0; i < 24; ++i)  {
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      input_ptr[i] = unif(rand_engine));
    }
    executor->sendLiterals();
    executor->executeComputation();
    delete [] input.buffer;

    float* output_ptr = (float*) output.buffer;
    //Check correctness
    for (int i = 0; i < 24; ++i)  {
      if (initializer.doubles()[i] > 0.0f)  {
        ONNX_ASSERT(almost_equal(initializer.floats()[i], output_ptr[i]));
      } else {
        ONNX_ASSERT(almost_equal(0.0f, output_ptr[i]));
      }
    }
    delete [] output.buffer;
  }
}

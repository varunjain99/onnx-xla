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
    initializer.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    std::uniform_real_distribution<float> unif(-0.5, 0.5);
    for (int i = 0; i < 24; ++i)  {
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      initializer.floats().push_back(unif(rand_engine));
    }
    initializer.sizes().push_back(2);
    initializer.sizes().push_back(3);
    initializer.sizes().push_back(4);
    std::vector<Dimension> sizes;
    sizes.push_back(2);
    sizes.push_back(3);
    sizes.push_back(4);
    relu_graph.addInitializerAndInput(initializer, "relu_input");
    auto relu_node = relu_graph.create(Symbol("Relu"), relu_graph.inputs());
    relu_graph.appendNode(relu_node);
    auto relu_output = relu_node->output();
    relu_output->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    relu_output->setSizes(sizes);
    relu_output->setUniqueName("relu_output");
    relu_graph.return_node()->addInput(relu_output);
    
    //Set up IO information
    uint64_t shape[3] = {2, 3, 4};
    onnxTensorDescriptor output;
    output.name = "relu_output";
    output.dataType = ONNXIFI_DATATYPE_FLOAT32;
    output.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    output.dimensions = 3;
    output.shape = shape;
    output.buffer = (onnxPointer) new float[24];
   
     //Execute using XLA backend
    XlaTransform runner(relu_graph, "relu");
    runner.translateGraph();
    auto executor = runner.executor();
    executor->initIO(0, nullptr, 1, &output);
    executor->sendLiterals();
    executor->executeComputation();

    std::cout << "returned from execution" << std::endl;    
    //Check correctness    
    float* output_ptr = (float*) output.buffer;
    for (int i = 0; i < 24; ++i)  {
      if (initializer.floats()[i] > 0.0f)  {
        ONNX_ASSERT(almost_equal(initializer.floats()[i], output_ptr[i]));
      } else {
        ONNX_ASSERT(almost_equal(0.0f, output_ptr[i]));
      }
    }
    delete [] output_ptr;
  }

  void dynamic_relu_test()  {
    //Set up IR graph
    Graph relu_graph;
    relu_graph.setName("relu_graph");
    Value* relu_input = relu_graph.addInput();
    std::vector<Dimension> sizes;
    sizes.push_back(2);
    sizes.push_back(3);
    sizes.push_back(4);    
    relu_input->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    relu_input->setSizes(sizes);
    relu_input->setUniqueName("relu_input");
    auto relu_node = relu_graph.create(Symbol("Relu"), relu_graph.inputs());
    relu_graph.appendNode(relu_node);
    auto relu_output = relu_node->output();
    relu_output->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    relu_output->setSizes(sizes);
    relu_output->setUniqueName("relu_output");
    relu_graph.return_node()->addInput(relu_output);
    
    //Set up IO information
    uint64_t shape[3] = {2, 3, 4};
    onnxTensorDescriptor output;
    onnxTensorDescriptor input;
    output.name = "relu_output";
    output.dataType = ONNXIFI_DATATYPE_FLOAT32;
    output.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    output.dimensions = 3;
    output.shape = shape;
    output.buffer = (onnxPointer) new float[24];
    input.name = "relu_input";
    input.dataType = ONNXIFI_DATATYPE_FLOAT32;
    input.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    input.dimensions = 3;
    input.shape = shape;
    input.buffer = (onnxPointer) new float[24];

    //Execute using XLA backend
    XlaTransform runner(relu_graph, "relu");
    runner.translateGraph();
    auto executor = runner.executor();
    executor->initIO(1, &input, 1, &output);
    float* input_ptr = (float*) input.buffer;
    std::uniform_real_distribution<float> unif(-0.5, 0.5);
    for (int i = 0; i < 24; ++i)  {
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      input_ptr[i] = unif(rand_engine);
    }
    executor->sendLiterals();
    executor->executeComputation();

    float* output_ptr = (float*) output.buffer;
    //Check correctness
    for (int i = 0; i < 24; ++i)  {
      if (input_ptr[i] > 0.0f)  {
        ONNX_ASSERT(almost_equal(input_ptr[i], output_ptr[i]));
      } else {
        ONNX_ASSERT(almost_equal(0.0f, output_ptr[i]));
      }
    }
    delete [] input_ptr;
    delete [] output_ptr;
  }
}

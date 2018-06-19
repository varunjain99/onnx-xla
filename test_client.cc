#include <grpcpp/grpcpp.h>
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

class XlaClient {
 public:
  XlaClient(std::shared_ptr<Channel> channel)
      : stub_(xla::grpc::XlaService::Stub::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string getDevice(void) {
    // Data we are sending to the server.
    ::xla::GetDeviceHandlesRequest request;
    request.set_device_count(1);

    // Container for the data we expect from the server.
    ::xla::GetDeviceHandlesResponse response;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.

    ::grpc::ServerContext context;

    // The actual RPC.
    Status status = stub_->GetDeviceHandles(&context, &request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return response.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  XlaClient client(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));
  std::string reply = client.getDevice();
  std::cout << "Client received: " << reply << std::endl;

  return 0;
}


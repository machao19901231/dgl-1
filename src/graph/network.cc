/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.cc
 * \brief DGL networking related APIs
 */

#include "./network/communicator.h"
#include "./network/socket_communicator.h"
#include "./network/serialize.h"

#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {
namespace network {

static char* global_data_buffer = nullptr;
const int kMaxBufferSize = 100 * 1024 * 1024;  // 100 MB

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string ip = args[0];
    int port = args[1];
    network::Communicator* comm = new network::SocketCommunicator();
    comm->Initialize(true, ip.c_str(), port);
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(comm);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    GraphHandle ghandle = args[1];
    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    auto csr = ptr->GetInCSR();  // We only care about in_csr
    const IdArray node_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    const IdArray edge_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[3]));
    const IdArray layer_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[4]));
    const IdArray flow_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[5]));
    char* data_buffer = nullptr;
    int64_t data_size = network::SerializeSampledSubgraph(
                             &data_buffer,
                             csr,
                             node_mapping,
                             edge_mapping,
                             layer_offsets,
                             flow_offsets);
    // Send msg via network
    int size = comm->Send(data_buffer, data_size);
    if (size <= 0) {
      LOG(ERROR) << "Send message erro (size: " << size << ")";
    }
    if (data_buffer != nullptr) {
      delete [] data_buffer;
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string ip = args[0];
    int port = args[1];
    int num_sender = args[2];
    int queue_size = args[3];
    network::Communicator* comm = new network::SocketCommunicator();
    comm->Initialize(false, ip.c_str(), port, num_sender, queue_size);
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(comm);
    global_data_buffer = new char[kMaxBufferSize];
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    int size = comm->Receive(global_data_buffer, kMaxBufferSize);
    if (size <= 0) {
      LOG(ERROR) << "Receive error: (size: " << size << ")";
    }
    NodeFlow* nf = new NodeFlow();
    ImmutableGraph::CSR::Ptr csr;
    network::DeserializeSampledSubgraph(global_data_buffer,
                                        &csr,
                                        &(nf->node_mapping),
                                        &(nf->edge_mapping),
                                        &(nf->layer_offsets),
                                        &(nf->flow_offsets));
    nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr, false));
    std::vector<NodeFlow*> subgs(1);
    subgs[0] = nf;
    *rv = WrapVectorReturn(subgs);
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeCommunicator")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    comm->Finalize();
    if (global_data_buffer != nullptr) {
      delete [] global_data_buffer;
      global_data_buffer = nullptr;
    }
  });

}  // namespace network
}  // namespace dgl

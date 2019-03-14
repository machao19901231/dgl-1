/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.h
 * \brief DGL networking related APIs
 */
#ifndef DGL_GRAPH_NETWORK_H_
#define DGL_GRAPH_NETWORK_H_

namespace dgl {
namespace network {

#define IS_SENDER   true
#define IS_RECEIVER false

// TODO(chao): make these number configurable

// Message cannot larger than 3GB
const int64_t kMaxBufferSize = 3221225472;
// Size of message queue is 5GB
const int64_t kQueueSize = 5368709120;

}  // namespace network
}  // namespace dgl
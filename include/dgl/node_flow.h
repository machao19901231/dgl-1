/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/node_flow.h
 * \brief DGL node flow class.
 */
#ifndef DGL_NODE_FLOW_H_
#define DGL_NODE_FLOW_H_

#include <string>
#include <vector>
#include <utility>
#include "graph_interface.h"

namespace dgl {

class NodeFlow {
 public:
  NodeFlow(GraphInterface::Ptr, IdArray, Surjection, Surjection);
  IdArray Front() const;
  IdArray Back() const;
  IdArray GetLayer(size_t i) const;
 private:
  // The graph structure.
  // It contains k NodeFlows, the right set of (i-1)^(th)
  //   NodeFlow and the left set of i^(th) NodeFlow are shared.
  GraphInterface::Ptr graph;
  // Offset of each layer.
  IdArray batch_offsets;
  // mapping to parent graph
  Surjection node_mapping;   // self id -> parent id
  Surjection edge_mapping;   // self id -> parent id
};

}  // namespace dgl

#endif  // DGL_NODE_FLOW_H_

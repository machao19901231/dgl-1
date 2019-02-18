import os
import dgl
import mxnet as mx

class Optimizer(object):
    def __init__(self, graph_name):
        self.multi_precision = False
        self.aggregate_num = 0
        self.initialized = {}
        self.weights = {}
        self.graph = dgl.DGLGraph(load_graph(graph_name), readonly=True)

    def create_state_multi_precision(self, index, weight):
        print("create multi-precision state")
        if isinstance(weight, mx.nd.sparse.RowSparseNDArray):
            data = mx.nd.ones(shape=weight.shape, dtype=weight.dtype)
            data.copyto(weight)
        self.weights.update({index : weight})
        return self

    def update_multi_precision(self, index, weight, grad, state):
        print("update multi-precision: " + index)
        # Update all node embeddings
        if index.startswith("dgl:update_all:layer-"):
            layer_id = get_layer_id(index)
            self.graph.update_all(self.msg_funcs[layer_id], self.reduce_funcs[layer_id],
                                  self.update_funcs[layer_id])
        # Update node embeddings
        elif isinstance(weight, mx.nd.sparse.RowSparseNDArray):
            #TODO this isn't a right way to copy data.
            grad.copyto(self.weights[index])
        else:
            # Update model weights.
            grad.copyto(self.weights[index])
        

class DGLKVstore(object):
    def __init__(self, embed_dict):
        self._kv = mx.kv.create('local')
        #self._kv = mx.kv.create('dist_sync')
        self._kv.set_optimizer(Optimizer())
        self._embed_dict = embed_dict
        for name in embed_dict:
            shape = embed_dict[name]['shape']
            dtype = embed_dict[name]['dtype']
            init_data = mx.nd.zeros(shape=(1, shape[1]), dtype=dtype)
            init_data = mx.nd.sparse.row_sparse_array((init_data, mx.nd.array([0], dtype='int64')),
                                                      shape=shape, dtype=dtype)
            self._kv.init(name, init_data)
            self._kv.push(name, init_data)

    def pull(self, name, row_ids):
        row_ids = mx.nd.array(row_ids, dtype='int64')
        data = mx.nd.sparse.zeros('row_sparse', shape=self._embed_dict[name]['shape'])
        self._kv.row_sparse_pull(name, row_ids=row_ids, out=data)
        return data.data

    def push(self, name, row_ids, embed):
        row_ids = mx.nd.array(row_ids, dtype='int64')
        self._kv.push(name, mx.nd.sparse.row_sparse_array((embed, row_ids),
                                                          shape=self._embed_dict[name]['shape'],
                                                          dtype=self._embed_dict[name]['dtype']))


import os
import sys
import dgl
import mxnet as mx
import pickle
import numpy as np
from dgl.data import load_data
from mxnet import gluon

if sys.version_info[0] > 2:
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    py_str = lambda x: x

class Arguments(object):
    def __init__(self, dataset):
        self.dataset = dataset

def load_graph(graph_name):
    args = Arguments(graph_name)
    data = load_data(args)
    return data.graph


def _remove_params(block):
    ret = {}
    params = block.collect_params()
    for name in params:
        param = params[name]
        ret.update({name : param._reduce()})
        param._data = None
        param._grad = None
    return ret


def _add_params(block, param_arrays):
    params = block.collect_params()
    for name in params:
        param = params[name]
        param._init_impl(param_arrays[name], param._ctx_list)


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
            self.graph.ndata[index] = weight
        self.weights.update({index : weight})
        return self

    def update_multi_precision(self, index, weight, grad, state):
        print("update multi-precision: " + index)
        # Update all node embeddings
        if index.startswith("dgl:update_all"):
            mfunc, rfunc, ufunc = pickle.loads(grad.asnumpy().tostring())
            if isinstance(mfunc, gluon.Block):
                _add_params(mfunc, self.weights)
            if isinstance(rfunc, gluon.Block):
                _add_params(rfunc, self.weights)
            if isinstance(ufunc, gluon.Block):
                _add_params(ufunc, self.weights)

            self.graph.update_all(mfunc, rfunc, ufunc)
        # Update node embeddings
        elif isinstance(weight, mx.nd.sparse.RowSparseNDArray):
            #TODO this isn't a right way to copy data.
            grad.copyto(self.weights[index])
        else:
            # Update model weights.
            grad.copyto(self.weights[index])
        

class DGLKVstore(object):
    def __init__(self, graph_name, embed_dict):
        self._kv = mx.kv.create('local')
        #self._kv = mx.kv.create('dist_sync')
        self._kv.set_optimizer(Optimizer(graph_name))
        self._embed_dict = embed_dict
        for name in embed_dict:
            shape = embed_dict[name]['shape']
            dtype = embed_dict[name]['dtype']
            init_data = mx.nd.zeros(shape=(1, shape[1]), dtype=dtype)
            init_data = mx.nd.sparse.row_sparse_array((init_data, mx.nd.array([0], dtype='int64')),
                                                      shape=shape, dtype=dtype)
            self._kv.init(name, init_data)
            self._kv.push(name, init_data)

        self._kv.init("dgl:update_all", mx.nd.zeros(shape=(1000), dtype=np.uint8))
        self._params = {}

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

    def _push_params(self, params):
        for name in params:
            if name not in self._params:
                self._kv.init(name, params[name])
                self._params.update({name : None})
            self._kv.push(name, params[name])

    def update_all(self, msg_func, reduce_func, update_func):
        if isinstance(msg_func, gluon.Block):
            msg_params = _remove_params(msg_func)
            self._push_params(msg_params)
        if isinstance(reduce_func, gluon.Block):
            reduce_params = _remove_params(reduce_func)
            self._push_params(reduce_params)
        if isinstance(update_func, gluon.Block):
            update_params = _remove_params(update_func)
            self._push_params(update_params)

        func_str = py_str(pickle.dumps((msg_func, reduce_func, update_func), 0))
        arr = np.fromstring(func_str, dtype=np.uint8)
        self._kv.push("dgl:update_all", mx.nd.array(arr, dtype=np.uint8))

        if isinstance(msg_func, gluon.Block):
            _add_params(msg_func, msg_params)
        if isinstance(reduce_func, gluon.Block):
            _add_params(reduce_func, reduce_params)
        if isinstance(update_func, gluon.Block):
            _add_params(update_func, update_params)

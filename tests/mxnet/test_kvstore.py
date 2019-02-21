import dgl
import numpy as np
import mxnet as mx
from mxnet import gluon
from dgl.data import load_data

class Arguments(object):
    def __init__(self, dataset):
        self.dataset = dataset

data = load_data(Arguments('cora'))
g = dgl.DGLGraph(data.graph)

in_feats = 10
out_feats = 10
embeddings = {'in' : {'shape' : (g.number_of_nodes(), in_feats), 'dtype' : 'float32'},
              'h' : {'shape' : (g.number_of_nodes(), out_feats), 'dtype' : 'float32'}}
kv = dgl.contrib.DGLKVstore('cora', embeddings)
data = mx.nd.random.normal(shape=(g.number_of_nodes(), in_feats))
kv.push('in', np.arange(0, data.shape[0], dtype=np.int64), data)
idx = [0, 2, 5]
assert np.all(kv.pull('in', idx).asnumpy() == data[idx].asnumpy())

def msg_func(edges):
    return {'m': edges.src['in']}

def reduce_func(nodes):
    return {'h' : mx.nd.sum(nodes.mailbox['m'])}

class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation='relu', bias=True):
        super(NodeUpdate, self).__init__()
        with self.name_scope():
            self.linear = gluon.nn.Dense(out_feats, use_bias=bias, activation=activation)

    def forward(self, node):
        return {'h': self.linear(node.data['h'])}

update = NodeUpdate(out_feats)
update.initialize(ctx=mx.cpu())
g.ndata['in'] = data
g.update_all(msg_func, reduce_func, update)
for _ in range(10):
    kv.update_all(msg_func, reduce_func, update)
assert np.all(kv.pull('h', np.arange(0, data.shape[0], dtype=np.int64)).asnumpy() == g.ndata['h'].asnumpy())

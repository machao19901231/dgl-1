import dgl
import mxnet as mx
from mxnet import gluon
from dgl.data import load_data

class Arguments(object):
    def __init__(self, dataset):
        self.dataset = dataset

data = load_data(Arguments('citeseer'))
g = dgl.DGLGraph(data.graph)

embeddings = {'h' : {'shape' : (3327, 10), 'dtype' : 'float32'}}
kv = dgl.contrib.DGLKVstore('citeseer', embeddings)
data = mx.nd.random.normal(shape=(5, 10))
kv.push('h', [0, 1, 2, 3, 4], data)
kv.push('h', [1, 2, 3, 4, 5], data)
print(kv.pull('h', [0, 2, 5]))
print(kv.pull('h', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

def msg_func(edges):
    return {'m': edges.src['h']}

def reduce_func(nodes):
    return {'h' : mx.nd.sum(nodes.mailbox['m'])}

class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeUpdate, self).__init__()
        with self.name_scope():
            self.linear = gluon.nn.Dense(out_feats, use_bias=bias, activation=activation)

    def forward(self, node):
        return {'h': self.linear(node.data['h'])}

update = NodeUpdate(10)
update.initialize(ctx=mx.cpu())
g.ndata['h'] = mx.nd.ones((g.number_of_nodes(), 10))
g.update_all(msg_func, reduce_func, update)
kv.update_all(msg_func, reduce_func, update)

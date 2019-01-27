import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
import dgl.function as fn
from dgl.data import register_data_args, load_data
import scipy as sp
from dgl import utils
from functools import partial


def sample_subgraph(g, seed_nodes, num_hops, num_neighbors):
    induced_nodes = []
    seeds = seed_nodes
    nodes_per_hop = [seeds]
    parent_uv_edges_per_hop = []
    for _ in range(num_hops):
        for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 1000000, num_neighbors,
                                                              neighbor_type='in',
                                                              seed_nodes=np.array(seeds),
                                                              return_seed_id=True):
            subg_src, subg_dst = subg.edges()
            parent_nid = subg.parent_nid
            src = parent_nid[subg_src]
            dst = parent_nid[subg_dst]
            parent_uv_edges_per_hop.append((src.asnumpy(), dst.asnumpy()))
            seeds = list(np.unique(src.asnumpy()))
            nodes_per_hop.append(seeds)
            induced_nodes.extend(list(parent_nid.asnumpy()))

    subgraph = g.subgraph(list(np.unique(np.array(induced_nodes))))
    subg_uv_edges_per_hop = [(subgraph.map_to_subgraph_nid(src).asnumpy(),
                              subgraph.map_to_subgraph_nid(dst).asnumpy())
                             for src, dst in parent_uv_edges_per_hop]
    return subgraph, subg_uv_edges_per_hop, nodes_per_hop


def gcn_msg(edge, ind, test=False):
    if test:
        msg = edge.src['h']
    else:
        msg = edge.src['h'] - edge.src['h_%d' % ind]
    return {'m': msg}


def gcn_reduce(node, ind, test=False):
    if test:
        accum = mx.nd.sum(node.mailbox['m'], 1) * node.data['deg_norm']
    else:
        accum = mx.nd.sum(node.mailbox['m'], 1) * node.data['norm'] + node.data['agg_h_%d' % ind] * node.data['deg_norm']
    return {'h': accum}


class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, dropout=0):
        super(NodeUpdate, self).__init__()
        self.linear = gluon.nn.Dense(out_feats, activation=activation)
        self.dropout = dropout

    def forward(self, node):
        accum = node.data['h']
        if self.dropout:
            accum = mx.nd.Dropout(accum, p=self.dropout)
        accum = self.linear(accum)
        return {'accum': accum}


class GCNLayer(gluon.Block):
    def __init__(self,
                 ind,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.ind = ind
        self.node_update = NodeUpdate(out_feats, activation, dropout)

    def forward(self, h, subg, subg_edges):
        subg.ndata['h'] = h
        if subg_edges is None:
            # test
            subg.update_all(partial(gcn_msg, ind=self.ind, test=True),
                            partial(gcn_reduce, ind=self.ind, test=True),
                            self.node_update)
        else:
            # control variate
            subg.send_and_recv(subg_edges, partial(gcn_msg, ind=self.ind),
                               partial(gcn_reduce, ind=self.ind),
                               self.node_update)

        h = subg.ndata.pop('accum')
        return h


class GCN(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.linear = gluon.nn.Dense(n_hidden, activation)
        self.layers = gluon.nn.Sequential()
        # hidden layers
        for i in range(1, n_layers):
            self.layers.add(GCNLayer(i, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.add(GCNLayer(n_layers, n_hidden, n_classes, None, dropout))


    def forward(self, subg, subg_edges_per_hop, nodes_per_hop):
        h = subg.ndata['preprocess']
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = self.linear(h)
        new_history = []
        for i, layer in enumerate(self.layers):
            if subg_edges_per_hop[0] is not None:
                indexes = subg.map_to_subgraph_nid(nodes_per_hop[self.n_layers-i])
                new_history.append(h.detach()[indexes])
            h = layer(h, subg, subg_edges_per_hop[self.n_layers-i-1])

        return h, new_history


def update_history(g, n_layers, new_history, nodes_per_hop):
    for i in range(n_layers):
        indexes = mx.nd.array(nodes_per_hop[n_layers-i]).astype('int64')
        hu = {'h_%d' % (i+1) : new_history[i]}
        g.set_n_repr(hu, indexes, inplace=True)


def evaluate(model, g, num_hops, labels, mask):
    pred, _ = model(g, [None for i in range(num_hops)], [None for i in range(num_hops)])
    pred = pred.argmax(axis=1)
    acc = ((pred == labels) * mask).sum().asscalar() / mask.sum().asscalar()
    print(acc)
    #print("Accuracy {:.4f}". format(acc))
    return acc


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.self_loop:
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_mask = mx.nd.array(data.train_mask)
    val_mask = mx.nd.array(data.val_mask)
    test_mask = mx.nd.array(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_nodes = data.graph.number_of_nodes()
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
              train_mask.sum().asscalar(),
              val_mask.sum().asscalar(),
              test_mask.sum().asscalar()))

    if args.gpu < 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        ctx = mx.gpu(args.gpu)

    num_neighbors = args.num_neighbors

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)
    # normalization
    degs = g.in_degrees().astype('float32')
    degs[degs > num_neighbors] = num_neighbors
    g.ndata['norm'] = mx.nd.expand_dims(1./degs, 1)
    deg_norm = mx.nd.expand_dims(1./g.in_degrees().astype('float32'), 1)
    g.ndata['deg_norm'] = deg_norm
    g.ndata['in'] = features
    g.update_all(fn.copy_src(src='in', out='m'),
                 fn.sum(msg='m', out='preprocess'))
    g.ndata['preprocess'] = g.ndata['preprocess'] * deg_norm

    num_data = len(train_mask)
    full_idx = np.arange(0, num_data)
    train_idx = full_idx[train_mask.asnumpy() == 1]

    seed_nodes = list(train_idx)
    num_hops = args.n_layers
    n_layers = args.n_layers
    n_hidden = args.n_hidden

    for i in range(n_layers):
        g.ndata['h_%d' % (i+1)] = mx.nd.zeros((features.shape[0], n_hidden))

    model = GCN(in_feats,
                n_hidden,
                n_classes,
                n_layers,
                'relu',
                args.dropout)
    model.initialize(ctx=ctx)
    n_train_samples = train_mask.sum().asscalar()
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
            {'learning_rate': args.lr, 'wd': args.weight_decay})

    # initialize graph
    dur = []
    test_acc_list = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()

        subg, subg_edges_per_hop, nodes_per_hop = sample_subgraph(g, seed_nodes, num_hops, num_neighbors)
        for i in range(n_layers):
            g.pull(nodes_per_hop[i], fn.copy_src(src='h_%d' % (i+1), out='m'),
                   fn.sum(msg='m', out='agg_h_%d' % (i+1)))

        subg_train_idx = subg.map_to_subgraph_nid(train_idx)
        subg.copy_from_parent()
        # forward
        subg_train_mask = np.zeros((len(subg.parent_nid),))
        subg_train_mask[subg_train_idx.asnumpy()] = 1

        with mx.autograd.record():
            pred, uh = model(subg, subg_edges_per_hop, nodes_per_hop)
            loss = loss_fcn(pred, labels[subg.parent_nid], mx.nd.expand_dims(mx.nd.array(subg_train_mask), 1))
            loss = loss.sum() / n_train_samples

        #print(loss.asnumpy())
        loss.backward()
        trainer.step(batch_size=1)
        update_history(g, n_layers, uh, nodes_per_hop)
        test_acc_list.append(evaluate(model, g, num_hops, labels, test_mask))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--num-neighbors", type=int, default=2,
            help="number of neighbors to be sampled")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)

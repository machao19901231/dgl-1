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
    def __init__(self, out_feats, activation=None, dropout=0, **kwargs):
        super(NodeUpdate, self).__init__(**kwargs)
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
                 bias=True, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.ind = ind
        self.node_update = NodeUpdate(out_feats, activation, dropout)

    def forward(self, h, subg, layer_nodes):
        subg.ndata['h'] = h
        if self.ind == 0 or layer_nodes is None:
            # first layer or test
            subg.update_all(partial(gcn_msg, ind=self.ind, test=True),
                            partial(gcn_reduce, ind=self.ind, test=True),
                            self.node_update)
        else:
            # control variate
            subg.pull(layer_nodes, partial(gcn_msg, ind=self.ind),
                      partial(gcn_reduce, ind=self.ind), self.node_update)
        h = subg.ndata.pop('accum')
        return h


class GCNForwardLayer(gluon.Block):
    def __init__(self,
                 ind,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True, **kwargs):
        super(GCNForwardLayer, self).__init__(**kwargs)
        self.ind = ind
        self.node_update = NodeUpdate(out_feats, activation, dropout)

    def forward(self, h, g):
        g.ndata['h'] = h
        # first layer or test
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        g.ndata['h'] = g.ndata['h'] * g.ndata['deg_norm']
        g.apply_nodes(self.node_update)
        agg_h = g.ndata.pop('h')
        h = g.ndata.pop('accum')
        return agg_h, h


class GCN(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.n_layers = n_layers
        with self.name_scope():
            #self.linear = gluon.nn.Dense(n_hidden, activation)
            self.layers = gluon.nn.Sequential()
            # input layer
            self.layers.add(GCNLayer(0, in_feats, n_hidden, activation, dropout))
            # hidden layers
            for i in range(1, n_layers-1):
                self.layers.add(GCNLayer(i, n_hidden, n_hidden, activation, dropout))
            # output layer
            self.layers.add(GCNLayer(n_layers-1, n_hidden, n_classes, None, dropout))


    def forward(self, subg, is_train):
        h = subg.ndata['in']
        new_history = []
        if is_train:
            for i, layer in enumerate(self.layers):
                indexes = subg.layer_nid(self.n_layers-i-1)
                h = layer(h, subg, indexes)
                # new history to be updated after one SGD iteration
                if i < self.n_layers:
                    new_history.append(h.detach().copy()[indexes])
            return h[indexes], new_history
        else:
            for i, layer in enumerate(self.layers):
                h = layer(h, subg, None)
            return h, None


class GCNUpdate(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout, **kwargs):
        super(GCNUpdate, self).__init__(**kwargs)
        self.n_layers = n_layers
        with self.name_scope():
            #self.linear = gluon.nn.Dense(n_hidden, activation)
            self.layers = gluon.nn.Sequential()
            # input layer
            self.layers.add(GCNForwardLayer(0, in_feats, n_hidden, activation, dropout))
            # hidden layers
            for i in range(1, n_layers-1):
                self.layers.add(GCNForwardLayer(i, n_hidden, n_hidden, activation, dropout))
            # output layer
            self.layers.add(GCNForwardLayer(n_layers-1, n_hidden, n_classes, None, dropout))


    def forward(self, g):
        h = g.ndata['in']
        agg_history = []
        for i, layer in enumerate(self.layers):
            agg_h, h = layer(h, g)
            agg_history.append(agg_h)
            g.ndata['agg_h_%d' % i] = agg_h
        return h


def evaluate(pred, num_hops, labels, mask):
    pred = pred.argmax(axis=1)
    accuracy = ((pred == labels) * mask).sum() / mask.sum().asscalar()
    acc = accuracy.asscalar()
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

    features = features.as_in_context(ctx)
    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    val_mask = val_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    num_neighbors = args.num_neighbors

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)
    # normalization
    degs = g.in_degrees().astype('float32')
    degs[degs > num_neighbors] = num_neighbors
    norm = mx.nd.expand_dims(1./degs, 1)
    if cuda:
        norm = norm.as_in_context(ctx)
    g.ndata['norm'] = norm
    g.ndata['deg_norm'] = mx.nd.expand_dims(1./g.in_degrees().astype('float32'), 1)
    g.ndata['in'] = features

    num_data = len(train_mask)
    full_idx = np.arange(0, num_data)
    train_idx = full_idx[train_mask.asnumpy() == 1]

    seed_nodes = list(train_idx)
    num_hops = args.n_layers + 1
    n_layers = args.n_layers
    n_hidden = args.n_hidden

    for i in range(n_layers):
        g.ndata['h_%d' % (i+1)] = mx.nd.zeros((features.shape[0], n_hidden))
        g.ndata['agg_h_%d' % (i+1)] = mx.nd.zeros((features.shape[0], n_hidden))

    model = GCN(in_feats,
                n_hidden,
                n_classes,
                n_layers,
                'relu',
                args.dropout, prefix='app')
    model.initialize(ctx=ctx)
    loss_fcn = gluon.loss.SoftmaxCELoss()
    model(g, False)

    update_model = GCNUpdate(in_feats,
                            n_hidden,
                            n_classes,
                            n_layers,
                            'relu',
                            args.dropout, prefix='app')
    update_model.initialize(ctx=ctx)
    update_model(g)

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
            {'learning_rate': args.lr, 'wd': args.weight_decay}, kvstore=mx.kv.create('local'))

    # initialize graph
    dur = []
    test_acc_list = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()

        for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 20, num_neighbors,
                                                              neighbor_type='in', num_hops=args.n_layers,
                                                              seed_nodes=np.array(seed_nodes),
                                                              return_seed_id=True):
            subg.copy_from_parent()
            # forward
            with mx.autograd.record():
                pred, uh = model(subg, True)
                loss = loss_fcn(pred, labels[subg.layer_parent_nid(0)])
                loss = loss.sum() / len(subg.layer_nid(0))

            #print(loss.asnumpy())
            loss.backward()
            trainer.step(batch_size=1)

        infer_params = update_model.collect_params()
        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())
        pred = update_model(g)

        test_acc_list.append(evaluate(pred, num_hops, labels, test_mask))


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

from multiprocessing import Process
import time
import numa
import argparse, time, math
import os
os.environ['DGLBACKEND'] = 'mxnet'
os.environ['OMP_NUM_THREADS'] = '8'
#os.environ['MXNET_CPU_PARALLEL_COPY_SIZE'] = '1000000000'
if os.environ['DMLC_ROLE'] == 'server':
    numa.bind([3])
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import profiler
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class NodeUpdate(gluon.Block):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            h = h * node.data['norm']
        else:
            h = h * node.data['sample_norm']
        h = self.dense(h)
        # skip connection
        if self.concat:
            h = mx.nd.concat(h, self.activation(h))
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}


class GCNSampling(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.dropout = dropout
        self.n_layers = n_layers
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            skip_start = (0 == n_layers-1)
            self.layers.add(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
            # hidden layers
            for i in range(1, n_layers):
                skip_start = (i == n_layers-1)
                self.layers.add(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
            # output layer
            self.layers.add(NodeUpdate(2*n_hidden, n_classes))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = mx.nd.Dropout(h, p=self.dropout)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             #lambda node : {'h': node.mailbox['m'].mean(axis=1)},
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h


class GCNInfer(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 **kwargs):
        super(GCNInfer, self).__init__(**kwargs)
        self.n_layers = n_layers
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            skip_start = (0 == n_layers-1)
            self.layers.add(NodeUpdate(in_feats, n_hidden, activation, test=True, concat=skip_start))
            # hidden layers
            for i in range(1, n_layers):
                skip_start = (i == n_layers-1)
                self.layers.add(NodeUpdate(n_hidden, n_hidden, activation, test=True, concat=skip_start))
            # output layer
            self.layers.add(NodeUpdate(2*n_hidden, n_classes, test=True))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             #lambda node : {'h': node.mailbox['m'].mean(axis=1)},
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h


def worker_func(worker_id, args, g, features, labels, train_mask, val_mask, test_mask,
                in_feats, n_classes, n_edges, train_nid, test_nid, n_test_samples):
    ctx = mx.cpu()
    profiler.set_config(profile_all=True, aggregate_stats=True,
            filename='profile_output-%d.json' % worker_id)
    numa.bind([worker_id % 3])
    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        mx.nd.relu,
                        args.dropout,
                        prefix='GCN')

    model.initialize(ctx=ctx)
    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           args.n_layers,
                           mx.nd.relu,
                           prefix='GCN')

    infer_model.initialize(ctx=ctx)

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create('dist_sync'))

    # initialize graph
    dur = []
    i = 0
    for epoch in range(args.n_epochs):
        start = time.time()
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                args.num_neighbors,
                neighbor_type='in',
                num_workers=32,
                shuffle=True,
                num_hops=args.n_layers+1,
                seed_nodes=train_nid):
            nf.copy_from_parent()
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1).astype('int64').as_in_context(ctx)
                batch_labels = labels[batch_nids]
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)
        train_time = time.time() - start

        infer_params = infer_model.collect_params()
        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.
        start = time.time()
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                args.test_num_neighbors,
                neighbor_type='in',
                num_workers=32,
                num_hops=args.n_layers+1,
                seed_nodes=test_nid):
            nf.copy_from_parent()
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1).astype('int64').as_in_context(ctx)
            batch_labels = labels[batch_nids]
            num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()
        infer_time = time.time() - start

        print(str(worker_id) + ": Test Accuracy {:.4f}, train: {:.3f}, infer: {:.3f}". format(num_acc/n_test_samples, train_time, infer_time))


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.gpu >= 0:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.Context('cpu_shared', 0)

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = mx.nd.array(np.nonzero(data.train_mask)[0]).astype(np.int64).as_in_context(ctx)
    test_nid = mx.nd.array(np.nonzero(data.test_mask)[0]).astype(np.int64).as_in_context(ctx)

    features = mx.nd.array(data.features).as_in_context(ctx)
    labels = mx.nd.array(data.labels).as_in_context(ctx)
    train_mask = mx.nd.array(data.train_mask).as_in_context(ctx)
    val_mask = mx.nd.array(data.val_mask).as_in_context(ctx)
    test_mask = mx.nd.array(data.test_mask).as_in_context(ctx)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    n_train_samples = train_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)

    g.ndata['features'] = features

    num_neighbors = args.num_neighbors

    degs = g.in_degrees().astype('float32').as_in_context(ctx)
    norm = mx.nd.expand_dims(1./degs, 1)
    g.ndata['norm'] = norm
    sample_degs = degs.asnumpy()
    sample_degs[sample_degs > num_neighbors] = num_neighbors
    sample_degs = mx.nd.array(sample_degs)
    sample_norm = mx.nd.expand_dims(1./sample_degs, 1)
    g.ndata['sample_norm'] = sample_norm.as_in_context(ctx)

    p1 = Process(target=worker_func, args=(0, args, g, features, labels, train_mask, val_mask, test_mask,
                                           in_feats, n_classes, n_edges, train_nid, test_nid, n_test_samples))
    p2 = Process(target=worker_func, args=(1, args, g, features, labels, train_mask, val_mask, test_mask,
                                           in_feats, n_classes, n_edges, train_nid, test_nid, n_test_samples))
    p3 = Process(target=worker_func, args=(2, args, g, features, labels, train_mask, val_mask, test_mask,
                                           in_feats, n_classes, n_edges, train_nid, test_nid, n_test_samples))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    print("parent ends")


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
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--test-num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--num-workers", type=int, default=1,
            help="The number of workers")
    args = parser.parse_args()

    print(args)

    main(args)


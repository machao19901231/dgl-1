import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import time

def main(args):
    # Start sender
    sender_train = dgl.contrib.sampling.SamplerSender(ip=args.ip, port=args.port)

    # load and preprocess dataset
    data = load_data(args)

    if args.gpu >= 0:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.cpu()

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = mx.nd.array(np.nonzero(data.train_mask)[0]).astype(np.int64).as_in_context(ctx)
    test_nid = mx.nd.array(np.nonzero(data.test_mask)[0]).astype(np.int64).as_in_context(ctx)

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)

    for epoch in range(args.n_epochs):
        # Here we onlt send nodeflow for training
        idx = 0
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
            print("send train nodeflow: %d" %(idx))
            sender_train.Send(nf)
            idx += 1
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    arser.add_argument("--ip", type=str, default='127.0.0.1',
            help="ip address of remote trainer machine")
    arser.add_argument("--port", type=int, default=2049,
            help="listen port of remote trainer machine")
    args = parser.parse_args()

    print(args)

    main(args)


import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def main(args):
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

    # sender_train = dgl.contrib.sampling.SamplerSender(ip='127.0.0.1', port=50051)
    # sender_infer = dgl.contrib.sampling.SamplerSender(ip='127.0.0.1', port=50052)

    for epoch in range(args.n_epochs):
        # Train sampler
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
            print("train")
            print(nf)
            # sender_train.Send(nf)

        # Infer sampler
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid):
            print("infer")
            print(nf)
            # sender_infer.Send(nf)
 


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
    args = parser.parse_args()

    print(args)

    main(args)


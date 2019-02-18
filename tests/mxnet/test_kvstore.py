import dgl
import mxnet as mx

embeddings = {'a' : {'shape' : (10, 10), 'dtype' : 'float32'}}
kv = dgl.contrib.DGLKVstore(embeddings)
data = mx.nd.random.normal(shape=(5, 10))
print(data)
kv.push('a', [0, 1, 2, 3, 4], data)
kv.push('a', [1, 2, 3, 4, 5], data)
print(kv.pull('a', [0, 2, 5]))
print(kv.pull('a', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

# Start 6 processes concurrently

# numa 0
DGLBACKEND=mxnet python3 sampler.py --ip 172.31.73.221 --port 2049 --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 &
DGLBACKEND=mxnet python3 sampler.py --ip 172.31.73.221 --port 2049 --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 &
# numa 1
DGLBACKEND=mxnet python3 sampler.py --ip 172.31.73.221 --port 2050 --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 &
DGLBACKEND=mxnet python3 sampler.py --ip 172.31.73.221 --port 2050 --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 &
# numa 2
DGLBACKEND=mxnet python3 sampler.py --ip 172.31.73.221 --port 2051 --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 &
DGLBACKEND=mxnet python3 sampler.py --ip 172.31.73.221 --port 2051 --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 &

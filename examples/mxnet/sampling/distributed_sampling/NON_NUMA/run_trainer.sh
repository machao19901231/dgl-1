DGLBACKEND=mxnet python3 trainer.py --ip 172.31.73.221 --port 2049 --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 --n-hidden 64
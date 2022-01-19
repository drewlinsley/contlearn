

cp netrc ../.netrc
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python run.py --config-name=WQ_synapses.yaml


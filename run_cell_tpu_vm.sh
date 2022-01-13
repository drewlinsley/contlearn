

export XRT_TPU_CONFIG="localservice;0;localhost:51011"

# pip3 install -r requirements.txts
cp netrc ../.netrc
# python3 run.py --config-name=celltype_gcp_tpu_1.yaml -m model.name=ResidualUNet2D
# export PT_XLA_DEBUG=1
# export USE_TORCH=ON
unset LD_PRELOAD

# DL the data
dlpath=$(python -c "from omegaconf import OmegaConf;conf = OmegaConf.load('conf/celltype_gcp_tpu_1.yaml');print(conf.data.path)")
gsutil cp $dlpath .

python3 run.py --config-name=celltype_gcp_tpu_1.yaml
# gsutil -m cp -r results/* gs://serrelab/connectomics/results/
# gsutil -m cp -r results/*/*/*/*/results/*/*/*/*/* gs://serrelab/connectomics/results/



export XRT_TPU_CONFIG="localservice;0;localhost:51011"

cp netrc ../.netrc
# export PT_XLA_DEBUG=1
# export USE_TORCH=ON

unset LD_PRELOAD

# python run.py --config-name=celltype_gcp_tpu_1.yaml
python run.py --config-name=WQ_muller_tpu.yaml
# gsutil -m cp -r results/* gs://serrelab/connectomics/results/
# gsutil -m cp -r results/*/*/*/*/results/*/*/*/*/* gs://serrelab/connectomics/results/

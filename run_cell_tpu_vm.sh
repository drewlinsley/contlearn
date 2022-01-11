

export XRT_TPU_CONFIG="localservice;0;localhost:51011"

# pip3 install -r requirements.txts
cp netrc ../.netrc
python3 run.py --config-name=celltype_gcp_tpu_1.yaml -m model.name=ResidualUNet2D
# gsutil -m cp -r results/* gs://serrelab/connectomics/results/
# gsutil -m cp -r results/*/*/*/*/results/*/*/*/*/* gs://serrelab/connectomics/results/

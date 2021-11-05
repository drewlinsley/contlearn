

export XRT_TPU_CONFIG="localservice;0;localhost:51011"

# pip3 install -r requirements.txts
python3 run.py --config-name=celltype_gcp_tpu_1.yaml -m model.name=UNet3D

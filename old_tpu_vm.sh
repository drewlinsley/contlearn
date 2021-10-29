

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

# pip3 install -r requirements.txts
python3 run.py --config-name=celltype_gcp.yaml -m model.name=UNet3D

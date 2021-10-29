
conda create --name gcp -y
conda activate gcp
conda install pathlib
cp netrc ../.netrc
pip install -r requirements.txt


export TPU_IP_ADDRESS=$(cat tpuip.txt)  # You could get the IP Address in the GCP TPUs section
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

# pip3 install -r requirements.txts
python3 run.py --config-name=celltype_gcp.yaml -m model.name=UNet3D

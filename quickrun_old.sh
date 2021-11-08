# quickrun_old.sh
. /anaconda3/etc/profile.d/conda.sh
conda activate torch-xla-1.10

export TPU_IP_ADDRESS=$(cat tpuip.txt)  # You could get the IP Address in the GCP TPUs section
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python3 run.py --config-name=celltype_gcp_tpu_1.yaml -m model.name=UNet3D

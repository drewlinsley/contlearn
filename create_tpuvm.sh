TPUNAME=pytorch-tpu1
# ZONE=us-central1-f  # europe-west4-a
ZONE=europe-west4-a
ZONE=us-east1-d
TPU=v3-256  # 8


gcloud alpha compute tpus tpu-vm delete $TPUNAME --zone=$ZONE --quiet
gcloud alpha compute tpus tpu-vm create $TPUNAME \
--zone=$ZONE \
--accelerator-type=$TPU \
--version=v2-alpha \
# --version=tpu-vm-pt-1.10  # v2-alpha \
# --boot-disk-size=200GB \

# gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE \
#   --command "git clone https://github.com/drewlinsley/contlearn.git && cd contlearn && pip3 -r install requirements.txt && bash tpu_vm.sh"
gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE \
  --command "git clone https://github.com/drewlinsley/contlearn.git && cd contlearn && git checkout gcp && sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 100 && cp netrc ../.netrc && pip install -r requirements.txt"
# gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE \
  # --command "cd contlearn && bash run_cell_tpu_vm.sh"
gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE
# 


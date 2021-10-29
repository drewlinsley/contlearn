VMNAME=pytorch
TPUNAME=pytorch-tpu
ZONE=us-central1-f  # europe-west4-a
# ZONE=europe-west4-a
# ZONE=us-east1-d
TPU=v3-8


###### DELETE
gcloud compute tpus execution-groups delete $TPUNAME \
--zone=$ZONE

##### CREATE
gcloud compute instances create $TPUNAME \
--zone=$ZONE  \
--machine-type=n1-standard-16  \
--image-family=torch-xla \
--image-project=ml-images  \
--boot-disk-size=200GB \
--scopes=https://www.googleapis.com/auth/cloud-platform
gcloud compute tpus create $TPUNAME \
--zone=$ZONE \
--network=default \
--version=pytorch-1.10 \
--accelerator-type=$TPU

##### GET TPU IP
TPUIP=$(gcloud compute tpus list | grep "$TPUNAME")
TPUIP=$(echo $TPUIP | cut -d "/" -f 1)
TPUIP=$(echo $TPUIP | cut -d " " -f 5)

##### CONNECT
gcloud compute ssh $TPUNAME --zone=$ZONE \
  --command "git clone https://github.com/drewlinsley/contlearn.git && cd contlearn  && git checkout gcp"
gcloud compute ssh $TPUNAME --zone=$ZONE \
  --command "echo $TPUIP >> tpuip.txt"
gcloud compute ssh $TPUNAME --zone=$ZONE \
  --command "sudo ln -s /anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && conda activate"
gcloud compute ssh $TPUNAME --zone=$ZONE \
  --command "cd contlearn && conda create --name gcp -y && conda activate gcp && conda install pathlib && cp netrc ../.netrc && pip install -r requirements.txt"
gcloud compute ssh $TPUNAME --zone=$ZONE


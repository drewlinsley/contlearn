VMNAME=pytorch
TPUNAME=pytorch-tpu
ZONE=us-central1-f  # europe-west4-a
# ZONE=europe-west4-a
# ZONE=us-east1-d
TPU=v3-8


###### DELETE
gcloud compute tpus execution-groups delete $VMNAME \
--zone=$ZONE
gcloud compute tpus delete $TPUNAME --zone=$ZONE --quiet

##### CREATE
gcloud compute instances create $VMNAME \
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
TPUIP=$(gcloud compute tpus describe $TPUNAME --zone=$ZONE | grep "\- ipAddress: ")
TPUIP=$(echo $TPUIP | cut -d ":" -f 2 | xargs)

##### CONNECT
gcloud compute ssh $VMNAME --zone=$ZONE \
  --command "git clone https://github.com/drewlinsley/contlearn.git && cd contlearn  && git checkout gcp"
gcloud compute ssh $VMNAME --zone=$ZONE \
  --command "echo $TPUIP >> contlearn/tpuip.txt"
# gcloud compute ssh $TPUNAME --zone=$ZONE \
#   --command "cd contlearn && "
gcloud compute ssh $VMNAME --zone=$ZONE

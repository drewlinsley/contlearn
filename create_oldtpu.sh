VMNAME=pytorch
TPUNAME=pytorch-tpu
ZONE=us-central1-f  # europe-west4-a
# ZONE=europe-west4-a
# ZONE=us-east1-d
TPU=v3-8


###### DELETE
gcloud compute instances delete $TPUNAME \
--zone=$ZONE  \
gcloud compute tpus delete $TPUNAME \
--zone=$ZONE \

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
--version=pytorch-1.9 \
--accelerator-type=$TPU

##### CONNECT
gcloud compute ssh $TPUNAME --zone=$ZONE \
  --command "git clone https://github.com/drewlinsley/contlearn.git && cd contlearn && pip install -r requirements.txt && git checkout gcp && cp netrc ../.netrc"
gcloud compute ssh $TPUNAME --zone=$ZONE


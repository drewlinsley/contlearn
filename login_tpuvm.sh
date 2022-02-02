ZONE=us-central1-f  # europe-west4-a us-east1-d
# ZONE=europe-west4-a  #  us-east1-d
TPU=v3-8  # 8

TPUNAME=$1

if [ -z "$TPUNAME" ]
then
  echo "Enter the name of your tpu on GCP:"
  read TPUNAME
  # pytorch-tpu1, muller-tpu-1
fi

gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE  --ssh-flag="-X"

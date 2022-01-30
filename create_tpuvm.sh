ZONE=us-central1-a  # europe-west4-a us-east1-d
TPU=v3-8  # 8

TPUNAME=$1
SCRIPT=$2

if [ -z "$TPUNAME" ]
then
  echo "Enter the name of your tpu on GCP:"
  read TPUNAME
  # pytorch-tpu1, muller-tpu-1
fi

if [ -z "$SCRIPT" ]
then
  echo "Enter the name of the script you want to run on GCP:"
  read SCRIPT
fi

if [ -f "$SCRIPT" ]; then
    echo "$SCRIPT exists."
else 
    echo "$SCRIPT does not exist, exiting."
    exit 0
fi


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

while True
do
  gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE \
    --command "cd contlearn && bash ${SCRIPT}"
done
# gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE  --ssh-flag="-X"

# ZONE=us-central1-a  # europe-west4-a us-east1-d
# ZONE=europe-west4-a  #  us-east1-d
TPU=v3-8   # 8

TPUNAME=$1
CONFIG=$2
ZONE=$3
CONFIGDIR=conf/

if [ -z "$TPUNAME" ]
then
  echo "Enter the name of your tpu on GCP:"
  read TPUNAME
  # pytorch-tpu1, muller-tpu-1
fi

if [ -z "$ZONE" ]
then
  ZONE=us-central1-a
fi


if [ -z "$CONFIG" ]
then
  echo "Enter the name of the config you want to run on GCP:"
  read CONFIG
fi

if [ -f "$CONFIGDIR$CONFIG" ]; then
    echo "Check for $CONFIG passed. Building TPU and running job."
else 
    echo "$CONFIGDIR$CONFIG does not exist, exiting."
    exit 0
fi


gcloud alpha compute tpus tpu-vm delete $TPUNAME --zone=$ZONE --quiet
gcloud alpha compute tpus tpu-vm create $TPUNAME --zone=$ZONE --accelerator-type=$TPU --version=v2-alpha
gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE \
  --command "git clone https://github.com/drewlinsley/contlearn.git && cd contlearn && git checkout gcp && sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 100 && cp netrc ../.netrc && pip install -r requirements.txt"

KEEPTRYING=true
while $KEEPTRYING
do
  STATUS=`gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE \
    --command "cd contlearn && bash scripts/run_tpu_vm_job.sh ${CONFIG}"`
  if grep -q "$Error" <<< "$STATUS"
  then
    echo $STATUS
    KEEPTRYING=true
  else
    KEEPTRYING=false
  fi
done

# Save the final ckpt
gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE \
  --command 'cd contlearn && FINALCKPT=`find . -name "*.ckpt" -type f | xargs ls -ltr | tail -1 | rev | cut -d" " -f1 | rev` && echo $FINALCKPT && gsutil cp $FINALCKPT gs://serrelab/connectomics/checkpoints/${CONFIG}.ckpt'

# gcloud alpha compute tpus tpu-vm ssh $TPUNAME --zone $ZONE
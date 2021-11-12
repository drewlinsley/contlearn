
# module load cuda/11.3.1
# module load cudnn/8.2.0
# module load gcc/10.2
# module load anaconda/3-5.2.0

# source deactivate
# source activate gcp


module load python/3.7.4
module load cuda/11.3.1
module load cudnn/8.2.0
virtualenv -p python3 pytorch.venv
source pytorch.venv/bin/activate
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html pathlib Pillow pytorch-lightning hydra-core wandb python-dotenv tensorflow tensorflow_datasets captum matplotlib torchvision tfrecord gcsfs pytest

python3 run.py --config-name=celltype_test.yaml -m model.name=UNet3D



module load python/3.8
module load cuda/11.3.1
module load cudnn/8.2.0
virtualenv -p python3 pytorch.venv
source pytorch.venv/bin/activate
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html pathlib Pillow pytorch-lightning hydra-core wandb python-dotenv tensorflow tensorflow_datasets captum matplotlib torchvision tfrecord gcsfs pytest


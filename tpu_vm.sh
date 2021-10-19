

export XRT_TPU_CONFIG="localservice;0;localhost:51011"

pip install -r requirements.txt
python run.py --config-name=celltype_gcp.yaml -m model.name=int

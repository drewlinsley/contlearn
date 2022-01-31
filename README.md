

# First prepare your data for model training
1. Install the other package
2. Download data (copy annotations to your account and set sharing to public if this is not already the case)
3. Create a config and run inference on your dataset
4. Upload to GCS or put in a local directory that can be accessed by this platform

# Create a TPU (VM)
\url{https://cloud.google.com/docs/authentication/getting-started}
install google cloud sdk
run init to set up your account

- examples

bash create_tpuvm.sh muller-tpu-1 conf/WQ_muller_tpu.yaml
bash create_tpuvm.sh synapse-tpu-1 conf/WQ_synapses_tpu.yaml
bash create_tpuvm.sh pytorch-tpu-1 conf/WQ_synapses_tpu.yaml
bash create_tpuvm.sh synapse-k0725-tpu-1 conf/k0725_synapses_tpu.yaml
bash create_tpuvm.sh bloodvessels-k0725-tpu-1 conf/k0725_bloodvessels_tpu.yaml
bash create_tpuvm.sh bpts-k0725-tpu-1 conf/k0725_bpts_tpu.yaml

# Run on GPU



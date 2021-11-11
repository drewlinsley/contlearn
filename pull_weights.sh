
echo "Enter gs path of weights"
read varname

echo $varname

mkdir weights
# gsutil -m cp gs://serrelab/connectomics/results/checkpoints/* checkpoints/
gsutil -m cp $varname checkpoints/


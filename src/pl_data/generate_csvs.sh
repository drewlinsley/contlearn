OUTDIR=src/pl_data/csvs/
mkdir $OUTDIR
gsutil ls gs://serrelab/connectomics/npzs/celltype/train/*.npz > $OUTDIR/celltype_train.csv
gsutil ls gs://serrelab/connectomics/npzs/celltype/val/*.npz > $OUTDIR/celltype_val.csv

#! /bin/bash

# Assumes each directory contains Jsonl files
input_data_dirs="/datadir/dataset1/ \
/datadir/dataset2/ \
/datadir/dataset3/"

output_dir="/outputdir/minhashes"

# NOTE: The script implicitly assumes that the last part
# of the input data paths is the dataset name and will choose
# output dir names as follows:
# /outputdir/minhashes/dataset1
# /outputdir/minhashes/dataset2
# /outputdir/minhashes/dataset3
# This can cause issues if the last part of the
# dirname is the same across datasets

mkdir -p $output_dir

# Is a good number for files 200MB or lesser
# Use a smaller value for larger jsonl files
files_per_partition=20

mkdir -p $output_dir
echo $input_data_dirs

python -u compute_minhashes.py \
  --input-data-dirs $input_data_dirs \
  --minhash-length 260 \
  --char-ngram 5 \
  --hash-bytes 4 \
  --seed 42 \
  --output-minhash-dir $output_dir \
  --log-dir $LOGDIR \
  --num-files $NUM_FILES \
  --files-per-partition $files_per_partition \
  --profile-path $PROFILESDIR \
  --log-frequency 250 \
  --scheduler-file $LOGDIR/scheduler.json

echo "Time Check: `date`"

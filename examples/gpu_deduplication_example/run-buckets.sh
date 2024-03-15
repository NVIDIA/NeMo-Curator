#!/bin/bash

minhash_dir="/outputdir/minhashes"
datasets=$(ls ${minhash_dir})
for dataset in $datasets; do
  input_minhash_dirs="$input_minhash_dirs $minhash_dir/$dataset/minhashes.parquet"
done
output_dir="/outputdir"

buckets_per_shuffle=1

mkdir -p $output_dir
echo $input_minhash_dirs

# Remove old buckets
rm -r ${output_dir}/buckets.parquet

python -u minhash_buckets.py \
  --input-data-dirs $input_minhash_dirs \
  --minhash-length 260 \
  --output-bucket-dir $output_dir/ \
  --log-dir $LOGDIR \
  --protocol ucx \
  --num-bands 20 \
  --buckets-per-shuffle=$buckets_per_shuffle \
  --split-out=512 \
  --scheduler-file $LOGDIR/scheduler.json

echo "Time Check: `date`"

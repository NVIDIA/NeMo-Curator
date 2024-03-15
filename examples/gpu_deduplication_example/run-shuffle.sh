input_data_dirs="/datadir/dataset1/ \
/datadir/dataset2/ \
/datadir/dataset3/"
buckets_dir="/outputdir/buckets.parquet"
output_dir="/outputdir"


export CUDF_SPILL="1"

## Run jaccard Mapping
echo "Starting Jaccard mapping..."
python jaccard_map_buckets.py \
  --input-bucket-dir $buckets_dir \
  --input-data-dirs $input_data_dirs \
  --output-dir $output_dir \
  --log-dir $LOGDIR \
  --text-ddf-blocksize 512 \
  --num-files $NUM_FILES \
  --scheduler-file $LOGDIR/scheduler.json

### Run jaccard Shuffle

echo "Starting Jaccard Shuffle..."

python jaccard_shuffle.py \
  --input-bucket-mapping-dir $output_dir/anchor_docs_with_bk.parquet \
  --input-data-dirs $input_data_dirs \
  --output-dir $output_dir \
  --text-ddf-blocksize 256 \
  --bucket-mapping-ddf-blocksize 512 \
  --num-files $NUM_FILES \
  --parts-per-worker 1 \
  --scheduler-file $LOGDIR/scheduler.json

echo "Time Check: `date`"

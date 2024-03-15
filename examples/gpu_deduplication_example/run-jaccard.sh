
shuffled_docs_dir="/outputdir/shuffled_docs.parquet"
output_dir="/outputdir"


export CUDF_SPILL="1"

python jaccard_compute.py \
  --shuffled-docs-path $shuffled_docs_dir \
  --output-dir $output_dir \
  --log-dir $LOGDIR \
  --num-files $NUM_FILES \
  --scheduler-file $LOGDIR/scheduler.json


echo "Time Check: `date`"

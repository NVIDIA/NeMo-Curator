
base_dir="/outputdir"
cc_folder="CC"
output_dir="${base_dir}/${cc_folder}_output"
cache_dir="${base_dir}/${cc_folder}_cache"
jaccard_pairs_path="/outputdir/dedup_final_results.parquet"


echo "output_dir set to $output_dir"
echo "cache_dir set to $cache_dir"

export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"

### compute connected component
#rm -r $cache_dir
mkdir -p $output_dir $cache_dir

python -u connected_component.py \
    --jaccard-pairs-path $jaccard_pairs_path \
    --output-dir $output_dir \
    --cache-dir $cache_dir \
    --log-dir $LOGDIR \
    --profile-path $PROFILESDIR \
    --num-files $NUM_FILES \
    --scheduler-file $LOGDIR/scheduler.json

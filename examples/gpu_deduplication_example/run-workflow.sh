#! /bin/bash

echo "Starting Workflow..."
echo "Time Check: `date`"
if [[ -z "$SLURM_JOB_ID" ]]; then
  TODAY="`date +"%Y_%m_%d"`"
else
  TODAY="`date +"%Y_%m_%d"`-$SLURM_JOB_ID"
fi

# Prepare output directory
export JOB_DIR=rapids-dedup-scripts/DEDUP-$TODAY
export FULL_OUTPUT_DIR=$HOME/$JOB_DIR
export LOGDIR=$FULL_OUTPUT_DIR/logs
export PROFILESDIR=$FULL_OUTPUT_DIR/profiles
# Take the default location within the container
RUNSCRIPT=${RUNSCRIPT:--/opt/nemo-data-curator/examples/gpu_deduplication_example/run-minhash.sh}
echo $RUNSCRIPT
mkdir -p $LOGDIR
mkdir -p $PROFILESDIR

cd /opt/nemo-data-curator/nemo_curator/gpu_deduplication
#-----#


# Env vars
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"

export LIBCUDF_CUFILE_POLICY=${LIBCUDF_CUFILE_POLICY:-ALWAYS}

# Network interface specific to the cluster being used
export INTERFACE=ibp12s0
export PROTOCOL=ucx
echo $INTERFACE

# This variable can be set to limit the number of jsonl files that
# are used in the dedup. Setting to -1 reads in all files
export NUM_FILES=-1

# Start the scheduler on the rank 0 node
if [[ -z "$SLURM_NODEID" ]] || [[ $SLURM_NODEID == 0 ]]; then
  echo "Starting scheduler"
  DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True \
  DASK_DISTRIBUTED__RMM__POOL_SIZE=1GB \
    dask scheduler \
      --scheduler-file $LOGDIR/scheduler.json \
      --protocol $PROTOCOL \
      --interface $INTERFACE >> $LOGDIR/scheduler.log 2>&1 &
fi
sleep 30

# Start the workers on each node
echo "Starting workers..."
dask-cuda-worker --scheduler-file $LOGDIR/scheduler.json --rmm-pool-size 72GiB --interface $INTERFACE --rmm-async >> $LOGDIR/worker_$HOSTNAME.log 2>&1 &

sleep 60

if [[ -z "$SLURM_NODEID" ]] || [[ $SLURM_NODEID == 0 ]]; then
  echo "Time Check: `date`"
  bash $RUNSCRIPT
  echo "Time Check: `date`"
  touch $LOGDIR/done.txt
fi

# All nodes wait until done
while [ ! -f $LOGDIR/done.txt ]
do
  sleep 15
done

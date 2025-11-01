#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Usage: $0 <experiment_name> <iteration> <dataset> <is_warmup> <target_model>" >&2
  exit 1
fi

EXPERIMENT_NAME=$1
ITERATION=$2
DATASET=$3
IS_WARMUP=$4
TARGET_MODEL=$5

if [ "${IS_WARMUP}" = "true" ]; then
  cd slurms_warmup
else
  cd slurms
fi

array_job_id=$(sbatch --parsable ${EXPERIMENT_NAME}.slurm ${EXPERIMENT_NAME} ${ITERATION} ${DATASET} ${TARGET_MODEL})
echo "Depending Array Job ID: $array_job_id"
sbatch --dependency=afterok:$array_job_id summarize.slurm ${TARGET_MODEL}_${EXPERIMENT_NAME}_${DATASET}_${ITERATION} ${DATASET}
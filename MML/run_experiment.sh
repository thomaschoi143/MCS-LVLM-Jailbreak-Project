#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <experiment_name> <iteration> <dataset>" >&2
  exit 1
fi

EXPERIMENT_NAME=$1
ITERATION=$2
DATASET=$3

cd slurms

array_job_id=$(sbatch --parsable ${EXPERIMENT_NAME}.slurm ${EXPERIMENT_NAME} ${ITERATION} ${DATASET})
echo "Submitted Job ID: $array_job_id"
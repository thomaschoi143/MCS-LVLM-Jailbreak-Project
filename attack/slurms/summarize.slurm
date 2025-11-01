#!/bin/bash

#SBATCH --partition=sapphire
#SBATCH --account="punim1943"

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

# SBATCH --mail-user=inchongc@student.unimelb.edu.au
# SBATCH --mail-type=FAIL
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END

#SBATCH --time=00:10:00

if [ $# -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <dataset>" >&2
    exit 1
fi

EXPERIMENT_NAME=$1
DATASET=$2

if [[ "$DATASET" == *"tiny"* ]]; then
  TASKS_INTERVAL=10
else
  TASKS_INTERVAL=12
fi

module purge
module load foss/2022a 
module load Python/3.10.4
cd ..
source ../.venvs/venv-3.10.4/bin/activate

python ../summarize_stats.py --attack_method=attack --experiment_name=${EXPERIMENT_NAME} --dataset=${DATASET} --tasks_interval=${TASKS_INTERVAL}

# Run the job from the directory where it was launched (default)

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
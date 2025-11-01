#!/bin/bash

# SBATCH --partition=gpu-a100
#SBATCH --partition=feit-gpu-a100
# SBATCH --partition=deeplearn
#SBATCH --account="punim1943"
#SBATCH --job-name="VRP"

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=51200

#SBATCH --gres=gpu:2
# SBATCH --qos=gpgpudeeplearn
#SBATCH --qos=feit
# SBATCH --constraint=dlg5

# SBATCH --mail-user=inchongc@student.unimelb.edu.au
# SBATCH --mail-type=FAIL
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END

#SBATCH --time=01:00:00

# Check if both arguments are provided
if [ $# -ne 3 ]; then
  echo "Usage: $0 <experiment_name> <iteration> <dataset>" >&2
  exit 1
fi

EXPERIMENT_NAME=$1
ITERATION=$2
DATASET=$3

PORT="20${ITERATION}0"

module purge
module load foss/2022a 
module load Python/3.10.4
cd ../../..
source .venvs/venv-3.10.4/bin/activate
cd attack
python server.py --dataset=${DATASET} --port=${PORT} --lvlm_name=qwen &

echo "Waiting for the server to start on port ${PORT}..."

while true; do
    if nc -z localhost ${PORT}; then
        echo "Server is running on port ${PORT}!"
        break
    else
        echo "Server not up yet. Checking again in 30 seconds..."
        sleep 30
    fi
done

cd ../visualRolePlay/query_specific
python inference.py --dataset=${DATASET} --experiment_name=${EXPERIMENT_NAME}_${DATASET}_${ITERATION} --port=${PORT}

# Run the job from the directory where it was launched (default)

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
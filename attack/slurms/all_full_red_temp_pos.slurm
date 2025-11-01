#!/bin/bash

# SBATCH --partition=gpu-a100
# SBATCH --partition=feit-gpu-a100
#SBATCH --partition=deeplearn
#SBATCH --account="punim1943"
#SBATCH --job-name="our"

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=51200

#SBATCH --gres=gpu:2
#SBATCH --qos=gpgpudeeplearn
# SBATCH --qos=feit
#SBATCH --constraint=dlg5

# SBATCH --mail-user=inchongc@student.unimelb.edu.au
# SBATCH --mail-type=FAIL
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END

#SBATCH --time=04:00:00

#SBATCH --array=0-20:10

# Check if both arguments are provided
if [ $# -ne 4 ]; then
  echo "Usage: $0 <experiment_name> <iteration> <dataset> <target_model>" >&2
  exit 1
fi

EXPERIMENT_NAME=$1
ITERATION=$2
DATASET=$3
TARGET_MODEL=$4

# Determine port suffix
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    PORT_SUFFIX="0"
else
    PORT_SUFFIX="${SLURM_ARRAY_TASK_ID:0:1}"
fi

if [ "$TARGET_MODEL" == "qwen" ]; then
    BASE_PORT="8"
elif [ "$TARGET_MODEL" == "llama" ]; then
    BASE_PORT="7"
else
    BASE_PORT="6"
fi

PORT="${BASE_PORT}8${ITERATION}${PORT_SUFFIX}"

module purge
module load foss/2022a 
module load Python/3.10.4
cd ../..
source .venvs/venv-3.10.4/bin/activate

cd attack
python server.py --load_clip --load_sd --load_sem_relevance --lvlm_name=${TARGET_MODEL} --port=${PORT} --dataset=${DATASET} &

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

python main.py --dataset=${DATASET} --task_i_start_from=${SLURM_ARRAY_TASK_ID} --num_tasks=12 --experiment_name=${TARGET_MODEL}_${EXPERIMENT_NAME}_${DATASET}_${ITERATION} --port=${PORT} --red_temperature=0.36

# Run the job from the directory where it was launched (default)

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
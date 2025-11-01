#!/bin/bash

#SBATCH --partition=feit-gpu-a100
#SBATCH --account="punim1943"
#SBATCH --job-name="MML_images"

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=51200

#SBATCH --gres=gpu:1
#SBATCH --qos=feit
# SBATCH --qos=gpgpudeeplearn
# SBATCH --constraint=dlg5

# SBATCH --mail-user=inchongc@student.unimelb.edu.au
# SBATCH --mail-type=FAIL
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END

#SBATCH --time=00:10:00

PORT=5010

module purge
module load foss/2022a 
module load Python/3.10.4
cd ..
source ../.venvs/venv-3.10.4/bin/activate

python ../attack/server.py --port=${PORT} &

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

python construct_images.py --dataset_name=harmbench --experiment_name=construct_images_harmbench --port=${PORT}
python construct_images.py --dataset_name=jailbreakbench --experiment_name=construct_images_jailbreakbench --port=${PORT}
# python construct_images.py --dataset_name=strongreject --experiment_name=construct_images_strongreject --port=${PORT}


# Run the job from the directory where it was launched (default)

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
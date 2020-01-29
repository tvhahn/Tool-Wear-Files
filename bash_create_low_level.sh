#!/bin/bash
#SBATCH --time=00:30:00 # 30 minutes
#SBATCH --array=1-92
#SBATCH --mem=1G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $
## How to use arrays: https://docs.computecanada.ca/wiki/Job_arrays

echo "Starting task $SLURM_ARRAY_TASK_ID"
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input_zip_files)

# Place the code to execute here
module load scipy-stack/2019b
python create_low_level.py $DIR

#!/bin/bash
#SBATCH --time=00:10:00 # 10 minutes
#SBATCH --mem=3G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $


echo "Starting task"

# Place the code to execute here
module load scipy-stack/2019b
python merge_temp_csv.py $DIR

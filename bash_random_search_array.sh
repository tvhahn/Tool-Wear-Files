#!/bin/bash
#SBATCH --time=01:00:00 # 1 hour
#SBATCH --array=1-100
#SBATCH --mem=4G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $
## How to use arrays: https://docs.computecanada.ca/wiki/Job_arrays

echo "Starting task $SLURM_ARRAY_TASK_ID"

# Place the code to execute here
module load scipy-stack/2019b
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index scikit_learn
pip install --no-index xgboost
# pip install /home/tvhahn/scikit_learn-0.22.1-cp37-cp37m-linux_x86_64.whl
pip install /home/tvhahn/imbalanced_learn-0.6.1-py3-none-any.whl
# pip install /home/tvhahn/xgboost-0.90-cp37-cp37m-linux_x86_64.whl

python random_search_run.py $SLURM_ARRAY_TASK_ID

#!/bin/bash
#SBATCH --time=00:10:00 # 10 minutes
#SBATCH --mem=1G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $


echo "Starting task"

# Place the code to execute here
module load scipy-stack/2019b
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
# pip install --no-index scikit_learn
# pip install --no-index xgboost
pip install /home/tvhahn/scikit_learn-0.22.1-cp37-cp37m-linux_x86_64.whl
pip install /home/tvhahn/imbalanced_learn-0.6.1-py3-none-any.whl
pip install /home/tvhahn/xgboost-0.90-cp37-cp37m-linux_x86_64.whl

python random_search_run.py 1

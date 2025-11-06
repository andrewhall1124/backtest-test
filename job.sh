#!/bin/bash

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16384M   # memory per CPU core
#SBATCH -J "Momentum Backtest"   # job name
#SBATCH --mail-user=amh1124@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source .venv/bin/activate
python main.py
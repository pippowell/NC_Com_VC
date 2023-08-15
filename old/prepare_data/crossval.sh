#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes 1
##SBATCH --mem 100G
#SBATCH -c 10
#SBATCH -p workq
#SBTACH --job-name SNN_Visual
#SBATCH --error=errors_crossval.o%j
#SBATCH --output=output_crossval.o%j

echo "running in shell: " "$SHELL"
echo "*** loading spack modules ***"

source ~/.bashrc
conda activate EEG_Vis_CL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/p/ppowell/miniconda3/envs/EEG_Vis_CL/lib/
echo $LD_LIBRARY_PATH

echo "*** set workdir ***"

/home/student/p/ppowell/miniconda3/envs/EEG_Vis_CL/bin/python create_crossval_splits.py "$@"

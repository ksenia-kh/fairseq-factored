#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --mem=8G # Memory
#SBATCH --ignore-pbs                                                            
####### --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/get_synsets_de_1.log

SCRIPT_PATH="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/babelfy/get_synsets.py"
PYTHON="python"

source ~/.bashrc
conda activate env

#stdbuf -i0 -e0 -o0 $PYTHON SCRIPT_PATH
$PYTHON SCRIPT_PATH > "/home/usuaris/veu/jordi.armengol/tfg/new/logs/get_synsets_de_1.log"
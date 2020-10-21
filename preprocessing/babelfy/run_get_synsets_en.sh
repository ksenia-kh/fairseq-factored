#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --mem=8G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/europarl_get_synsets_en.log

SCRIPT_PATH="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/preprocessing/babelfy/get_synsets_en.py"
PYTHON="python"

source ~/.bashrc
conda activate myenv

stdbuf -i0 -e0 -o0 $PYTHON $SCRIPT_PATH
#$PYTHON SSCRIPT_PATH > "/home/usuaris/veu/jordi.armengol/tfg/new/logs/get_synsets_de_1.log"

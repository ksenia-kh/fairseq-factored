#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-attn-fr.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
CP="checkpoint_last.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
OUTPUT_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings_attn/fr"
N=100
#mkdir tmp

# Activate conda environment
source ~/.bashrc
conda activate myenv
git checkout attention-analysis


mkdir -p  $OUTPUT_DIR

: '
#Bilingual fr baseline
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints33-fr"
OUTPUT="extracted-attn-enfr-b.pkl"
SRC="en_tokensS"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe
'

# Factored fr POS
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints34-fr-pos-new"
OUTPUT="extracted-attn-enfr-f-pos.pkl"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --multiple-encoders False #--remove-bpe

# Factored de TAGS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-tags"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints35-fr-tags-new"
OUTPUT="extracted-attn-enfr-f-tags.pkl"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --multiple-encoders False #--remove-bpe


# Factored de DEPS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-deps"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints36-fr-deps-new"
OUTPUT="extracted-attn-enfr-f-deps.pkl"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --multiple-encoders False #--remove-bpe


# Factored de Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints37-fr-l"
OUTPUT="extracted-attn-enfr-f-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --multiple-encoders False #--remove-bpe


# Factored de Syn-Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-syn-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints38-fr-syn"
OUTPUT="extracted-attn-enfr-f-syn-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --multiple-encoders False #--remove-bpe


# Factored de Syn-Pos

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-syn-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints39-fr-syn-pos"
OUTPUT="extracted-attn-enfr-f-syn-pos.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --multiple-encoders False #--remove-bpe



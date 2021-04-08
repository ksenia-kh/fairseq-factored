#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-attn-es.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"
CP="checkpoint_last.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
OUTPUT_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings_attn/es"
N=100
#mkdir tmp

# Activate conda environment
source ~/.bashrc
conda activate myenv
git checkout attention-analysis


mkdir -p  $OUTPUT_DIR


#Bilingual es baseline
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/baseline"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints7-es-b"
OUTPUT="extracted-attn-enes-b.pkl"
SRC="en_tokensS"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe


: '
# Factored es POS
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints8-es-pos"
OUTPUT="extracted-attn-enes-f-pos.pkl"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False

# Factored es TAGS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-tags"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints9-es-tags"
OUTPUT="extracted-attn-enes-f-tags.pkl"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es DEPS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-deps"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints10-es-deps"
OUTPUT="extracted-attn-enes-f-deps.pkl"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints11-es-l"
OUTPUT="extracted-attn-enes-f-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es Syn-Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-syn-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints12-es-syn"
OUTPUT="extracted-attn-enes-f-syn-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es Syn-Pos

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-syn-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints25-es-syn-pos"
OUTPUT="extracted-attn-enes-f-syn-pos.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False
'


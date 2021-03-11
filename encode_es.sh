#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-es.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"
CP="checkpoint_last.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
OUTPUT_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/es"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-es/"
N=3888
#mkdir tmp

# Activate conda environment
source ~/.bashrc
conda activate myenv


mkdir -p  $OUTPUT_DIR

#Bilingual es baseline
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints7-es-b"
OUTPUT="encodings-enes-b.json"
SRC="en_tokensS"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe

# Factored es POS
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints8-es-pos"
OUTPUT="encodings-enes-f-pos.json"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False

# Factored es TAGS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints9-es-tags"
OUTPUT="encodings-enes-f-tags.json"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es DEPS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints10-es-deps"
OUTPUT="encodings-enes-f-deps.json"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints11-es-l"
OUTPUT="encodings-enes-f-lemmas.json"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es Syn-Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints12-es-syn"
OUTPUT="encodings-enes-f-syn-lemmas.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored es Syn-Pos

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints25-es-syn-pos"
OUTPUT="encodings-enes-f-syn-pos.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="es_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False



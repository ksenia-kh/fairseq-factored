#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-fr.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
CP="checkpoint_last.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"
N=3888
#mkdir tmp



encode-lstm() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    
    head --l $N  $INPUT_DATA > tmp/tmp.lstm.$SRC 
    
    sed  -i 's/^/<2es> /' tmp/tmp.lstm.$SRC

    #subword-nmt apply-bpe -c $DEST_DIR/corpus.tc.shuf.codes < tmp/tmp.$SRC > tmp/tmp.bpe.$SRC

    
    CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
        --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
        --newkey $SRC-$TGT --newarch lstm --output-file $OUTPUT \
        --newtask translation --remove-bpe < tmp/tmp.lstm.$SRC 
}

encode() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    
    head --l $N  $INPUT_DATA > tmp/tmp.$SRC 
    
    sed  -i 's/^/<2es> /' tmp/tmp.$SRC

    #subword-nmt apply-bpe -c $DEST_DIR/corpus.tc.shuf.codes < tmp/tmp.$src > tmp/tmp.bpe.$src

    
    cuda_visible_devices="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
        --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
        --newkey $SRC-$TGT --newarch transformer_big --output-file $OUTPUT \
        --newtask translation --remove-bpe < tmp/tmp.$SRC 
}

encode_notag() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    
    #head --l $N  $INPUT_DATA > tmp/tmp.$SRC
    

    #subword-nmt apply-bpe -c $dest_dir/corpus.tc.shuf.codes < tmp/tmp.$src > tmp/tmp.bpe.$src

    
    cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --output-file $OUTPUT --remove-bpe < tmp/tmp.$SRC
}



encode-interlingua() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4

    head --l $N  $INPUT_DATA > tmp/tmp.$SRC 

    #subword-nmt apply-bpe -c $DEST_DIR/codes/$SRC.codes < tmp/tmp.$SRC > tmp/tmp.bpe.$SRC


    CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
     --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
     --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
     --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
     --newkey $SRC-$TGT --newarch multilingual_transformer  --output-file $OUTPUT  \
     --newtask multilingual_translation --remove-bpe < tmp/tmp.$SRC 

}
# Activate conda environment
source ~/.bashrc
conda activate myenv


mkdir -p  /home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr

#Bilingual FR baseline
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints33-fr"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr/"
OUTPUT="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr/encodings-enfr-b.json"
SRC="en_tokensS"
TGT="fr_tokensS"

#cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
#         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
#        --output-file $OUTPUT --remove-bpe

# Factored FR POS
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints34-fr-pos-new"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr"
OUTPUT="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr/encodings-enfr-f-pos.json"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT --remove-bpe --multiple-encoders False

# Factored FR TAGS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints35-fr-tags-new"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr"
OUTPUT="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr/encodings-enfr-f-tags.json"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT --remove-bpe --multiple-encoders False


# Factored FR DEPS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints36-fr-deps-new"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr"
OUTPUT="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr/encodings-enfr-f-deps.json"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT --remove-bpe --multiple-encoders False


# Factored Fr Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints37-fr-l"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr"
OUTPUT="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr/encodings-enfr-f-lemmas.json"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT --remove-bpe --multiple-encoders False


# Factored Fr Syn-Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints38-fr-syn"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr"
OUTPUT="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr/encodings-enfr-f-syn-lemmas.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT --remove-bpe --multiple-encoders False


# Factored Fr Syn-Pos

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints39-fr-syn-pos"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr"
OUTPUT="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr/encodings-enfr-f-syn-pos.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="fr_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT --remove-bpe --multiple-encoders False



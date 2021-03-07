#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/interactive-encode-fr-baseline.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
CP="checkpoint_last.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"
N=3888
mkdir tmp



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
    
    head --l $N  $INPUT_DATA > tmp/tmp.$SRC 
    

    #subword-nmt apply-bpe -c $dest_dir/corpus.tc.shuf.codes < tmp/tmp.$src > tmp/tmp.bpe.$src

    
    cuda_visible_devices="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
        --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
        --newkey $SRC-$TGT --newarch transformer_wmt_en_de --output-file $OUTPUT --newtask translation --remove-bpe < tmp/tmp.$SRC
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
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
encode_notag /home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr/mt.bpe.en_tokensS /tfm/data/mt/encodings/fr/encodings-enfr-b.json en_tokensS fr_tokensS





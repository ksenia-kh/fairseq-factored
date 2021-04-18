
seed=$(($RANDOM));

FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
ENCODINGS_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/ru"
VOCABULARY="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-preprocessed-bpe/dict.en_tokensS.txt"
LABELS="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en.txt"

# Activate conda environment
source ~/.bashrc
conda activate myenv

echo $seed

echo ''
echo 'Baseline En-Ru'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enru-b.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Ru POS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enru-f-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Ru TAGS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enru-f-tags.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Ru DEPS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enru-f-deps.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Ru Lemmas'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enru-f-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Ru Syn Lemmas'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enru-f-syn-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Ru Syn Pos'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enru-f-syn-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed
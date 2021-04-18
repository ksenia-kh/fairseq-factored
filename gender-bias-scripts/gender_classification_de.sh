
seed=$(($RANDOM));

FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
ENCODINGS_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/de"
VOCABULARY="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe/dict.en_tokensS.txt"
LABELS="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en.txt"

# Activate conda environment
source ~/.bashrc
conda activate myenv

echo $seed

echo ''
echo 'Baseline En-De'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-ende-b.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-De POS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-ende-f-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-De TAGS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-ende-f-tags.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-De DEPS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-ende-f-deps.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-De Lemmas'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-ende-f-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-De Syn Lemmas'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-ende-f-syn-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-De Syn Pos'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-ende-f-syn-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

seed=$(($RANDOM));

echo $seed

echo ''
echo 'Language Specific'
echo '************************************************'
echo ''
python gender_classification.py -e encodings-inter.json \
    -v ~/interlingua-nodistance/data-bin/europarl/dict.en.txt \
    -l ../raw_data/en.txt \
    -o 0 \
    -s $seed


echo ''
echo 'Language Specific with Russian'
echo '************************************************'
echo ''
python gender_classification.py -e encodings-inter-ru.json \
    -v ~/interlingua-nodistance/data-bin/europarl/dict.en.txt \
    -l ../raw_data/en.txt \
    -o 0 \
    -s $seed


echo ''
echo 'Shared' 
echo '************************************************'
echo ''

python gender_classification.py -e encodings-shared-noru.json  \
    -v /home/usuaris/veu/cescola/fairseq/data-bin/multi-europarl-noru-joint/dict.src.txt \
    -l ../raw_data/en.txt \
    -o 1 \
    -s $seed



echo ''
echo 'Shared with Russian' 
echo '************************************************'
echo ''

python gender_classification.py -e encodings-shared.json  \
    -v /home/usuaris/veu/cescola/fairseq/data-bin/multi-europarl-ru-joint/dict.src.txt \
    -l ../raw_data/en.txt \
    -o 1 \
    -s $seed


echo ''
echo 'Bilingual ENFR'
echo '************************************************'
echo ''

python gender_classification.py -e encodings-enfr2.json \
    -v ~/interlingua-nodistance/data-bin/europarl/dict.en.txt \
    -l ../raw_data/en.txt \
    -o 0 \
    -s $seed

echo ''
echo 'Bilingual ENDE'
echo '************************************************'
echo ''

python gender_classification.py -e encodings-ende.json \
    -v ../bilingual/en-de/dict.en.txt \
    -l ../raw_data/en.txt \
    -o 0 \
    -s $seed

echo ''
echo 'Bilingual ENES'
echo '************************************************'
echo ''

python gender_classification.py -e encodings-enes.json \
    -v ../bilingual/en-es/dict.en.txt \
    -l ../raw_data/en.txt \
    -o 0 \
    -s $seed

echo ''
echo 'Bilingual ENRU'
echo '************************************************'
echo ''

python gender_classification.py -e encodings-enru.json \
    -v ../bilingual/en-ru/dict.en.txt \
    -l ../raw_data/en.txt \
    -o 0 \
    -s $seed



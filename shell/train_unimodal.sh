#!/usr/bin/env bash
for i in {1800..2000..10} 
do 
FVAR=$(find data/Google_ngram/ -name "$i*");
echo $FVAR
mkdir -p models/Google_ngram/unimodal/$i/
python word2gm_trainer.py --num_mixtures 1 --train_data $FVAR --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/Google_ngram/unimodal/$i/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 2
done
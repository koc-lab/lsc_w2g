#!/usr/bin/env bash

# Modify data and model folders accordingly
# English
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/en-semeval/c1.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/EnglishModels/c1/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/en-semeval/c2.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/EnglishModels/c2/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10

# German
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/de-semeval/c1.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/GermanModels/c1/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/de-semeval/c2.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/GermanModels/c2/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10

# Latin
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/ln-semeval/c1.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/LatinModels/c1/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/ln-semeval/c2.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/LatinModels/c2/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10

# Swedish
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/sw-semeval/c1.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/SwedishModels/c1/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10
python word2gm_trainer.py --num_mixtures 1 --train_data data/semEval/sw-semeval/c2.txt --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path models/SemEval/SwedishModels/c2/ --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10

# Semantic Change Detection with Gaussian Word Embeddings

In this study, we propose a Gaussian word embedding (w2g) based methodology and present a comprehensive study for the LSC detection.W2g is a probabilistic distribution-based word embedding model and represents words as Gaussian mixture models using covariance information along with the existing mean (word vector) [1]. We also extensively study several aspects of w2g-based LSC detection under the SemEval-2020 Task 1 evaluation framework, [2], as well as using Google Books N-gram Fiction corpus, [3].

## Preliminary

In order to start using our model, one should refer to the following repositories:

* [Word2GM (Word to Gaussian Mixture)](https://github.com/benathi/word2gm): This repository contains w2g implementation for the used word embedding. 

* [N-gram Word2Vec](https://github.com/ziyin-dl/ngram-word2vec): This repository contains tensorflow n-gram kernel for word embeddings and it is also useful for scraping Google N-gram Fiction corpus. This repository is necesssary only when Google N-gram Fiction corpus is used during training.

## Corpora

During this study, we used two different datasets to detect Lexical Semantic Change. These datasets were:

* [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/): By clicking on this link, one can reach corpora and ground-truth results for SemEval-2020 Task 1. For more details one can refer to [2]. 

* [Google Books N-gram](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html): This dataset contains n-grams in various categories. It is used extensively in LSC detection studies. In order to download this dataset, we advise the usage of [N-gram Word2Vec](https://github.com/ziyin-dl/ngram-word2vec) since this repository also contains source code to divide n-grams by decades.

## Training and Evaluation

Training and evaluation steps for SemEval and Google N-gram are different however in order to reduce complexity, we provided necessary source code and bash script.

### SemEval 2020 Task 1

In our study, we provide results for the models trained without using stop words. In order to obtain *stopless* SemEval corpora one should run the following:

```
python SemEval_Clean_Corpora.py
```

The remaining part of the code is set to regular corpora selection. However by changing data folders to the *stopless* corpora, one can use the same shell script. To train the model:

```
bash train_semeval.sh
```

During the evaluation process, our model require frequent word list. To generate this list run the following code:

```
python SemEval_FW_List.py
```

Results will be generated for SemEval as a result of this code:

```
python SemEval_Generate_Scores.py --corpora Normal --anchor common --th_method Gamma
```

We provided various settings for the evaluation code which are also visible in our paper:

```
optional arguments:
  -h, --help            show this help message and exit
  --corpora CORPORA     Corpus Types Options: Normal, Stopless
  --anchor ANCHOR       Shared Vocabulary Options: common, frequent
  --th_method TH_METHOD
                        Threshold Method, Options: Local Gamma, Custom. Local correponds to Language-Specific setting. In custom setting, users  
                        provide threshold values. Should be used if Cross-Language threshold is applied.
  --fw_ratio FW_RATIO
  -l TH_LIST [TH_LIST ...], --th_list TH_LIST [TH_LIST ...]
                        Threshold List necessary for Custom Threshold. Must be a list of 2 floats.
```


### Google N-gram

During n-gram training, word2vec kernel of w2g code should be adjusted accordingly. For this purpose, replace [Word2GM's (Word to Gaussian Mixture)](https://github.com/benathi/word2gm) word2vec_kernels.cc with the word2vec_kernels.cc of [N-gram Word2Vec](https://github.com/ziyin-dl/ngram-word2vec). Compile the C skipgram as advised in Word2GM implementation:

```
chmod +x compile_skipgram_ops.sh
./compile_skipgram_ops.sh
```

We provided two shell scripts, the first one can be used to train single mode gaussians (Unimodal), the second script can be used to analyze multimodal word embeddings in the context of LSC:


```
bash train_unimodal.sh
bash train_bimodal.sh
```

In order to visualize Google N-gram w2g models, Google_ngram_visualization should be used:

```
python Google_ngram_visualization.py
```

## Requirements
* tensorflow 
* pandas
* numpy
* scipy
* collections
* matplotlib
* ggplot
* nltk

## Citation

This part will be added when our article is published.

## References

[1] Athiwaratkun, B., & Wilson, A. (2017). Multimodal  word  distributions. In Proc. of the 55th Annual Meeting of the Assoc. for Comp. Ling. (Volume 1: Long  Papers)(pp. 1645–1656).

[2] D. Schlechtweg, B. McGillivray, S. Hengchen, H. Dubossarsky, and N. Tahmasebi, “SemEval-2020 Task 1: Unsupervised lexical semantic change detection,” in Proc. of the 14th Int. Workshop on Semantic Evaluation. Barcelona, Spain: Assoc. for Comp. Ling. (ACL), 2020.

[3] Y. Lin, J.-B. Michel, E. Aiden Lieberman, J. Orwant, W. Brockman, and S. Petrov, “Syntactic annotations for the Google Books Ngram corpus,” in Proc. of the ACL 2012 System Demonstrations. Jeju Island, Korea: Assoc. for Comp. Ling. (ACL), Jul. 2012, pp. 169–174.


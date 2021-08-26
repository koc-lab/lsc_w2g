from collections import defaultdict
from collections import Counter
#from quantitative_eval import *
import matplotlib.pyplot as plt
import pandas as pd
from word2gm_loader import Word2GM
import numpy as np

# Vocab Files

# Change to stopless if stopless FW lists are composed
en_C1Path = "models/normal/EnglishModels/c1Unimodal/"
en_C2Path = "models/normal/EnglishModels/c2Unimodal/"
de_C1Path = "models/normal/GermanModels/c1Unimodal/"
de_C2Path = "models/normal/GermanModels/c2Unimodal/"
sw_C1Path = "models/normal/SwedishModels/c1Unimodal/"
sw_C2Path = "models/normal/SwedishModels/c2Unimodal/"
ln_C1Path = "models/normal/LatinModels/c1Unimodal/"
ln_C2Path = "models/normal/LatinModels/c2Unimodal/"


lang = ["en", "de", "sw", "lt"]
paths = [en_C1Path, en_C2Path, de_C1Path, de_C2Path,
         sw_C1Path, sw_C2Path, ln_C1Path, ln_C2Path]


for l in lang:
    frequentWords = defaultdict(list)
    c1Path = paths[0]
    c2Path = paths[1]
    paths = paths[2:]
    TXT = "vocab.txt"
    w2gm = Word2GM(c1Path)
    w2gm2 = Word2GM(c2Path)
    list_temp = set(w2gm.id2word) & set(w2gm2.id2word)
    common_words = sorted(list_temp, key=lambda k: w2gm2.id2word.index(k))
    print("lang is : " + l)
    print("Reading File for C1: " + c1Path)
    print("Reading File for C2: " + c2Path)
    c1Path = c1Path + TXT
    c2Path = c2Path + TXT
    for ratio in np.arange(0.05, 1.05, 0.05):
        print("Ratio is : " + str(ratio))
        frequentWords = defaultdict(list)
        fYear = Counter()
        fYear += Counter()
        for filePath in [c1Path, c2Path]:
            with open(filePath) as f:
                for line in f:
                    line = line.split()
                    fYear[line[0]] = float(line[1])
            total = sum(fYear.values(), 0.0)
            length = int(ratio*len(fYear.values()))
            for word, word_count in fYear.most_common(length):
                frequentWords[word].append(
                    word_count/total)  # Normalized Frequency

        for key in frequentWords.keys():
            if(key in common_words and len(frequentWords[key]) == 2):
                region = float(frequentWords[key][1]) / \
                    float(frequentWords[key][0])
            else:
                frequentWords.pop(key)

        dataf = pd.DataFrame.from_dict(frequentWords, orient="index")
        dataf.columns = ["c1", "c2"]
        dataf.to_csv("FWLists/semeval_"+str(ratio)+"_"+l + "_FW.csv")
        dataf.to_pickle("FWLists/semeval_"+str(ratio)+"_"+l + "_FW.pkl")

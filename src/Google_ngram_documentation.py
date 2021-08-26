
from collections import defaultdict
from collections import Counter
from word2gm_loader import Word2GM
from quantitative_eval import *
import matplotlib.pyplot as plt
import pandas as pd
dataPath = "/models/GoogleNgram/unimodal_v2/"
yearList = range(1900, 2010, 10)
fileName = "/vocab.txt"

commonWords = defaultdict(list)
for year in yearList:
    filePath = dataPath + str(year) + fileName
    print("Reading File: " + filePath)
    with open(filePath) as f:
        for line in f:
            line = line.split()
            commonWords[line[0]].append(line[1])
    for key in commonWords.keys():
        if(len(commonWords[key]) != (year-1900)/10 + 1):
            commonWords.pop(key)

count = 0
for key in commonWords.keys():
    if(len(commonWords[key]) != (2000-1900)/10 + 1):
        count = count + 1

print(count)
print(len(commonWords.keys()))
dataf = pd.DataFrame.from_dict(commonWords, orient="index")
dataf.columns = yearList
dataf.to_csv("CommonWordsAfter1900.csv")

frequentWords = defaultdict(list)

for year in yearList:
    filePath = dataPath + str(year) + fileName
    print("Reading File: " + filePath)
    fYear = Counter()
    fYear += Counter()
    with open(filePath) as f:
        for line in f:
            line = line.split()
            fYear[line[0]] = float(line[1])
    total = sum(fYear.values(), 0.0)
    length = int(0.15*len(fYear.values()))
    for word, word_count in fYear.most_common(length):
        frequentWords[word].append(word_count/total)  # Normalized Frequency

    for key in frequentWords.keys():
        if(len(frequentWords[key]) != (year-1900)/10 + 1):
            frequentWords.pop(key)
        elif(len(frequentWords[key]) > 1):
            region = float(frequentWords[key][-1]) / \
                float(frequentWords[key][-2])
            if(region > 1.7 or region < 0.6):
                frequentWords.pop(key)

print(len(frequentWords.keys()))

dataf = pd.DataFrame.from_dict(frequentWords, orient="index")
dataf.columns = yearList
dataf.to_csv("FrequentWordsAfter1900.csv")


target_words = ["gay", "nice", "record"]
years = range(1800, 1970, 10)
freq_words = ["british", "mankind", "theatre"]
target_dict = defaultdict(list)
freq_dict = defaultdict(list)
target_dict_1 = defaultdict(list)
freq_dict_1 = defaultdict(list)

biPath = "models/GoogleNgram/bimodal/"
uniPath = "models/GoogleNgram/unimodal/"
saveUni = "results/GoogleNgram/unimodal/"
saveBi = "results/GoogleNgram/bimodal/"
for year in years:
    #model_dir_uni = uniPath + str(year) + "/"
    model_dir_bi = biPath + str(year) + "/"
    #w2gm_uni = Word2GM(model_dir_uni)
    w2gm_bi = Word2GM(model_dir_bi)
    print(str(year))
    print("-------------------------------------------")

    """
    # In order to print additional information, uncomment this section
    print "Selected Frequent Words"
    for word in freq_words:
      print( "For " + word)
      print "Unimodal" 
      w2gm_uni.show_nearest_neighbors(word, 0, saveUni, "freq" + str(year))
      print "Bimodal" 
      w2gm_bi.show_nearest_neighbors(word, 0, saveBi, "freq" + str(year))
      w2gm_bi.show_nearest_neighbors(word, 1, saveBi, "freq" + str(year))
    print "Selected Target Words"
    """
    temp = w2gm_bi.getMixes(target_words, 0)
    temp_1 = w2gm_bi.getMixes(freq_words, 0)
    temp_2 = w2gm_bi.getMixes(target_words, 1)
    temp_3 = w2gm_bi.getMixes(freq_words, 1)
    for i in range(0, len(target_words)):
        target_dict[target_words[i]].append(temp[i])
        target_dict_1[target_words[i]].append(temp_2[i])
        freq_dict[freq_words[i]].append(temp_1[i])
        freq_dict_1[freq_words[i]].append(temp_3[i])
    """ 
  # Visualize nearest neighbors
  for word in target_words:
    print( "For " + word)
    print "Unimodal" 
    w2gm_uni.show_nearest_neighbors(word, 0, saveUni, "target" + str(year))
    print "Bimodal"
    w2gm_bi.show_nearest_neighbors(word, 0, saveBi, "target" + str(year))
    w2gm_bi.show_nearest_neighbors(word, 1, saveBi, "target" + str(year))
  print "-----------------------------------------------------------------------"
  print "\n"
  """
for key in target_dict.keys():
    plt.clf()
    plt.plot(years, target_dict[key])
    plt.ylabel('Variances')
    plt.xlabel('Decades')
    plt.title('Variance Decade Plot for target word ={} for sense 0'.format(key))
    plotname = saveUni + "bimodal_target_0" + key + ".png"
    plt.savefig(plotname)

for key in freq_dict.keys():
    plt.clf()
    plt.plot(years, freq_dict[key])
    plt.ylabel('Variances')
    plt.xlabel('Decades')
    plt.title('Variance Decade Plot for frequent word ={} for sense 0'.format(key))
    plotname = saveUni + "bimodal_freq_0" + key + ".png"
    plt.savefig(plotname)

for key in target_dict_1.keys():
    plt.clf()
    plt.plot(years, target_dict_1[key])
    plt.ylabel('Variances')
    plt.xlabel('Decades')
    plt.title('Variance Decade Plot for target word ={} for sense 1'.format(key))
    plotname = saveUni + "bimodal_target_1" + key + ".png"
    plt.savefig(plotname)

for key in freq_dict_1.keys():
    plt.clf()
    plt.plot(years, freq_dict_1[key])
    plt.ylabel('Variances')
    plt.xlabel('Decades')
    plt.title('Variance Decade Plot for frequent word ={} for sense 1'.format(key))
    plotname = saveUni + "bimodal_freq_1" + key + ".png"
    plt.savefig(plotname)

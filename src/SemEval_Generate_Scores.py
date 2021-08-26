import argparse
import numpy as np
import scipy
import pandas as pd

import nltk
from nltk.corpus import stopwords
from scipy.stats import spearmanr

from word2gm_loader import Word2GM
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt

# Initilization
w2gm = ''
w2gm2 = ''

# Language Choices:
lang = ["en", "de", "sw", "lt"]

# Folder names can be used for further use
# For simplicity, the folder name declerations are added to top

semeval_data_folder = "data/SemEval/"
# Binary Ground Truth Folders
binary_gt = {
    'en': semeval_data_folder + 'English/truth/binary.txt',
    'de': semeval_data_folder + 'German/truth/binary.txt',
    'sw': semeval_data_folder + 'Swedish/truth/binary.txt',
    'lt': semeval_data_folder + 'Latin/truth/binary.txt',
}

# Graded Ground Truth Folders
graded_gt = {
    'en': semeval_data_folder + 'English/truth/graded.txt',
    'de': semeval_data_folder + 'German/truth/graded.txt',
    'sw': semeval_data_folder + 'Swedish/truth/graded.txt',
    'lt': semeval_data_folder + 'Latin/truth/graded.txt',
}

# Model Folders
stopless_model_folder = "models/stopless/"
normal_model_folder = "models/normal/"
corpora_keys = ['en_c1', 'en_c2', 'de_c1',
                'de_c2', 'sw_c1', 'sw_c2', 'lt_c1', 'lt_c2']

stopless_folders, normal_folders = {}, {}
for c in corpora_keys:
    stopless_folders[c] = stopless_model_folder + c
    normal_folders[c] = model_folder + c


''' 
Find kl divergence between Gaussians
Taken from Athiwaratkun's implementation
'''


def kl(w2gm, w2gm2, w1, cl1):
    w2 = w1
    cl2 = cl1
    # This is for KL and min KL
    # This is -2*KL(w1 || w2)
    D = len(w2gm.mus_n_multi[0, 0])
    # note: ignore -D because it's a constant
    m1 = w2gm.mus_n[w2gm.get_idx(w1)]
    m2 = w2gm2.mus_n[w2gm2.get_idx(w2)]
    epsilon = 1e-4
    logsig1 = w2gm.logsigs[w2gm.get_idx(w1)]
    logsig2 = w2gm2.logsigs[w2gm2.get_idx(w2)]
    sig1 = np.exp(logsig1)
    sig2 = np.exp(logsig2)
    s2_inv = 1./(epsilon + sig2)

    sph = (len(logsig1) == 1)

    # print 'D = {} Spherical = {}'.format(D, sph)

    diff = m1 - m2
    exp_term = np.sum(diff*s2_inv*diff)

    if sph:
        tr_term = D*sig1*s2_inv
    else:
        tr_term = np.sum(sig1*s2_inv)

    if sph:
        log_rel_det = D*logsig1 - D*logsig2
    else:
        log_rel_det = np.sum(logsig1 - logsig2)

    res = tr_term + exp_term - log_rel_det
    return res


''' 
Get Mean for first w2g
'''


def get_mu_unimodal_w2gm(word):
    id = w2gm.get_idx(word)
    cl = 0
    return w2gm.mus_n[id*w2gm.num_mixtures + cl]


''' 
Get Mean for second w2g
'''


def get_mu_unimodal_w2gm2(word):
    id = w2gm2.get_idx(word)
    cl = 0
    return w2gm2.mus_n[id*w2gm2.num_mixtures + cl]


'''
This function gets the frequent words in two time periods
'''


def get_frequent_words(lang='en', ratio=0.05):
    pickleFile = "data/FWLists/semeval_" + str(ratio) + "_" + lang + "_FW.pkl"
    fwords_df = pd.read_pickle(pickleFile)
    x = fwords_df.index.tolist()
    list_temp = set(w2gm.id2word) & set(w2gm2.id2word)
    setX = set(x).intersection(list_temp)

    y = []
    '''
    # This part removes stop words if necessary
    # Latin StopWords are taken from: https://wiki.digitalclassicist.org/Stopwords_for_Greek_and_Latin
    if( lang == 'en'):  
      y = list(stopwords.words('english'))
    elif(lang == 'de'):
      y = list(stopwords.words('german'))  
    elif(lang == 'sw'):
      y = list(stopwords.words('swedish')) 
    else:
      y = ["ab", "ac", "ad", "adhic", "aliqui", "aliquis", "an", "ante", "apud", "at", "atque", "aut", "autem", "cum", "cur", "de", "deinde", "dum", "ego", "enim", "ergo", "es", "est", "et", "etiam", "etsi", "ex", "fio", "haud", "hic", "iam", "idem", "igitur", "ille", "in", "infra", "inter", "interim", "ipse", "is", "ita", "magis", "modo", "mox", "nam",               "ne", "nec", "necque", "neque", "nisi", "non", "nos", "o", "ob", "per", "possum", "post", "pro", "quae", "quam", "quare", "qui", "quia", "quicumque", "quidem", "quilibet",              "quis", "quisnam", "quisquam", "quisque", "quisquis", "quo", "quoniam", "sed", "si", "sic", "sive", "sub", "sui", "sum", "super", "suus", "tam", "tamen", "trans", "tu", "tum", "ubi","uel", "uero"]

    '''
    res = [item for item in setX if item not in y and item.isalpha()]
    res = sorted(res, key=lambda k: w2gm2.word2id[k])
    return res


'''
This function finds the common words for two time periods
'''


def find_common_words(lang='en'):
    list_temp = set(w2gm.id2word) & set(w2gm2.id2word)
    #temp_index = [ w2gm2.word2id(w) for w in temp_list ]
    # sorted(list_temp, key = lambda k : w2gm.wordindex(k))
    common_words = list_temp

    y = []
    '''
    # This part removes stop words if necessary
    # Latin StopWords are taken from: https://wiki.digitalclassicist.org/Stopwords_for_Greek_and_Latin
    if( lang == 'en'):  
      y = list(stopwords.words('english'))
    elif(lang == 'de'):
      y = list(stopwords.words('german'))  
    elif(lang == 'sw'):
      y = list(stopwords.words('swedish')) 
    else:
      y = ["ab", "ac", "ad", "adhic", "aliqui", "aliquis", "an", "ante", "apud", "at", "atque", "aut", "autem", "cum", "cur", "de", "deinde", "dum", "ego", "enim", "ergo", "es", "est",             "et", "etiam", "etsi", "ex", "fio", "haud", "hic", "iam", "idem", "igitur", "ille", "in", "infra", "inter", "interim", "ipse", "is", "ita", "magis", "modo", "mox", "nam",               "ne", "nec", "necque", "neque", "nisi", "non", "nos", "o", "ob", "per", "possum", "post", "pro", "quae", "quam", "quare", "qui", "quia", "quicumque", "quidem", "quilibet",              "quis", "quisnam", "quisquam", "quisque", "quisquis", "quo", "quoniam", "sed", "si", "sic", "sive", "sub", "sui", "sum", "super", "suus", "tam", "tamen", "trans", "tu",                 "tum", "ubi","uel", "uero"]
    '''
    r = [item for item in common_words if item not in y and item.isalpha()]

    res = sorted(r, key=lambda k: w2gm2.word2id[k])
    return res


'''
This function aligns two embeddins based on the shared vocabulary settings
'''


def intersection_alignment(lang='en', setting='common', ratio=0.05, target_words=[]):
    if(setting == 'common'):
        common_words = find_common_words(lang)
    else:
        common_words = get_frequent_words(lang, ratio)
    #common_words = list(set(common_words)-set(target_words))
    newvectors1 = np.empty(
        (len(common_words), get_mu_unimodal_w2gm(common_words[0]).shape[0]))
    newvectors2 = np.empty(
        (len(common_words), get_mu_unimodal_w2gm2(common_words[0]).shape[0]))
    for i in range(len(common_words)):
        newvectors1[i] = get_mu_unimodal_w2gm(common_words[i])
        newvectors2[i] = get_mu_unimodal_w2gm2(common_words[i])
    return (newvectors1, newvectors2, common_words)


'''
This function uses OPM to align embeddings
'''


def procrustes_alignment(lang='en', setting='common', ratio=0.05, target_words=[]):
    global w2gm
    global w2gm2
    (new_vectors1, new_vectors2, common_words) = intersection_alignment(
        lang, setting, ratio, target_words)
    r, scale = scipy.linalg.orthogonal_procrustes(new_vectors2, new_vectors1)
    w2gm2.mus_n = np.dot(w2gm2.mus_n, r)
    return common_words


'''
This function finds the distance values for the target words
'''


def FindDistance(target_words, shared_words):
    global w2gm  # c1
    global w2gm2  # c2

    jdRes = []
    cdRes = []
    minkl1, minkl2, maxkl2, maxkl1 = 10000, 10000, 0, 0

    # For scaling the KL divergence values to the cosine distance range
    for w in shared_words:
        a = kl(w2gm, w2gm2, w, 0)[0][0]
        b = kl(w2gm, w2gm2, w, 0)[0][0]
        minkl1 = min(minkl1, a)
        maxkl1 = max(maxkl1, a)
        minkl2 = min(minkl2, b)
        maxkl2 = max(maxkl2, b)

    # Using target words cosine distance and jeffreys divergence values are calculated
    for i in range(len(target_words)):
        word = target_words[i]
        #print("The cosine distance between base vector and transformed vector")
        result_cd = scipy.spatial.distance.cosine(
            get_mu_unimodal_w2gm(word), get_mu_unimodal_w2gm2(word))

        resultkl = 0.5*(kl(w2gm2, w2gm, word, 0)[0][0]-minkl2)/(maxkl2-minkl2)
        resultkl2 = 0.5*(kl(w2gm, w2gm2, word, 0)[0][0]-minkl1)/(maxkl1-minkl1)
        result_jd = resultkl + resultkl2
        jdRes.append(result_jd)
        cdRes.append(result_cd)
    # Returns cosine and jeffreys array
    return np.array(cdRes), np.array(jdRes)


'''
This function finds the spearman correlation for the calculated answers and ground truth
'''


def getSpearman(answer, truth):
    spearCorr, spearP = spearmanr(answer, truth)
    return spearCorr


def GetTargetWords(lang='en'):
    res = {}
    file = open(binary_gt[lang], 'r')
    for line in file:
        words = line.split('\t')
        words[1] = int(words[1][0])
        res[words[0]] = []
        res[words[0]].append(words[1])
    file.close()

    file = open(graded_gt[lang], 'r')
    for line in file:
        words = line.split('\t')
        words[1] = float(words[1])
        res[words[0]].append(words[1])
    file.close()
    # Returns Target Words and Binary Grading
    return res


def accuracy(l1, l2):
    pos = 0
    for i in range(len(l1)):
        if(l1[i] == l2[i]):
            pos += 1
    return float(pos)/float(len(l1))


def Scores_SemEval(embedding='Normal', sharedVocab='common', thresholdMethod='Local', ratio=0.05, threshold=[]):
    global w2gm  # c1
    global w2gm2  # c2
    print("------------------------------------------------------")
    custom_thres = threshold
    csv_path = "MeanData/"
    if(sharedVocab == 'common'):
        csv_path += 'CW_'
        print("Common \n")
    else:
        csv_path += 'FW_'
        print("Frequent \n")

    if(embedding == 'Normal'):
        csv_path += 'N_'
        print("Normal Models \n")
        folders = normal_folders
    else:
        csv_path += 'S_'
        print("Stopless Models \n")
        folders = stopless_folders

    all_df = pd.DataFrame(columns=["Words", "Binary", "Graded", "CD"])

    #eng, de, sw, lt, avg
    spearman_cd_final = []
    binary_cd_final = []

    spearman_jd_final = []
    binary_jd_final = []
    # Iterate for each language
    for l in lang:

        # Load W2G models
        key_c1 = l + "_c1"
        key_c2 = l + "_c2"
        c1_dir = folders[key_c1]
        c2_dir = folders[key_c2]
        w2gm = Word2GM(c1_dir)
        w2gm2 = Word2GM(c2_dir)

        # Get Target Words
        binary_res = GetTargetWords(l)
        target_words = binary_res.keys()  # Target Words will be the keys of the dictionary
        ground_truth = np.array(binary_res.values()).T
        binary_truth = ground_truth[0].astype(int)
        graded_truth = ground_truth[1]

        # Align W2G models
        shared_words = procrustes_alignment(
            l, sharedVocab, ratio, target_words)

        # This part is for finding means from shared vocab
        cdRes, jdRes = FindDistance(shared_words, shared_words)

        '''
        # Store Results
        df_temp = pd.DataFrame(columns= ["Words", "Res", "CD"])
        df['Words'] = binary_res.keys()
        df['Res'] = binary_res.values()
        df = pd.DataFrame(df_temp["Res"].to_list(), columns=['Binary', 'Graded'])
        print(df)

        # This part is for finding means from shared vocab
        # Comment it in the original run

        df = pd.DataFrame(columns= ["Words", "CD", "jd"])
        df['Words'] = shared_words

        df['CD'] = cdRes        

        df['jd'] = jdRes
        all_df['Words'] += binary_res.keys()
        all_df['Binary'] += binary_res.values()
        all_df['CD'] += cdRes
        all_df['jd'] += jdRes

        all_df = pd.concat([all_df, df])
        '''
        # Find Accuracy and Spearman
        print("Language = {} \n".format(l))

        if(thresholdMethod == 'Local'):  # Language-Specific
            threshold = [np.mean(cdRes), np.mean(jdRes)]
        elif(thresholdMethod == 'Gamma'):
            threshold = [mquantiles(cdRes)[2], 0]
        elif(thresholdMethod == 'Custom'):  # Used for Cross-Language
            threshold = [custom_thres[0][lang.index(
                l)], custom_thres[1][lang.index(l)]]

        # Binary Classification for CD and JD
        cd_bin = np.where(cdRes >= threshold[0], 1, 0)
        jd_bin = np.where(jdRes >= threshold[1], 1, 0)
        cd_acc = 100*accuracy(binary_truth, cd_bin)
        jd_acc = 100*accuracy(binary_truth, jd_bin)

        # Spearman Correlation for CD and JD
        cd_spearman = getSpearman(graded_truth, cdRes)
        jd_spearman = getSpearman(graded_truth, jdRes)

        # Add results to the array
        binary_cd_final.append(cd_acc)
        spearman_cd_final.append(cd_spearman)
        binary_jd_final.append(jd_acc)
        spearman_jd_final.append(jd_spearman)

        print("Cosine Distance Accuracy = {:.2f} \n".format(cd_acc))
        print("Cosine Distance Spearman = {:.2f}  \n".format(cd_spearman))
        print("Jeffreys Divergence Accuracy = {:.2f} \n".format(jd_acc))
        print("Jeffreys Divergence Spearman = {:.2f}  \n".format(jd_spearman))

    binary_cd_final.append(np.mean(binary_cd_final))
    spearman_cd_final.append(np.mean(spearman_cd_final))
    binary_jd_final.append(np.mean(binary_jd_final))
    spearman_jd_final.append(np.mean(spearman_jd_final))
    return binary_cd_final, spearman_cd_final, binary_jd_final, spearman_jd_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--corpora', help='Corpus Types Options: Normal, Stopless')
    parser.add_argument(
        '--anchor', help='Shared Vocabulary Options: common, frequent')
    parser.add_argument(
        '--th_method', help='Threshold Method, Options: Local Gamma, Custom. Local correponds to Language-Specific setting. In custom setting, users provide threshold values. Should be used if Cross-Language threshold is applied.')
    parser.add_argument('--fw_ratio', type=int)

    parser.add_argument('-l', '--th_list', nargs='+', type=float,
                        help='Threshold List necessary for Custom Threshold. Must be a list of 2 floats.', required=True)

    args = parser.parse_args()
    if args.anchor == 'CW':
        args.fw_ratio = "100"
    if args.th_method != "Custom":
        args.th_list = []
    print("Current Custimization Details \nCorpora : {}, Shared Vocabulary: {}, Threshold Method: {}, FW_ratio: {} ".format(
        args.corpora, args.anchor, args.th_method, args.fw_ratio))
    Scores_SemEval(embedding=args.corpora, sharedVocab=args.anchor,
                   thresholdMethod=args.th_method/100, ratio=args.fw_ratio, threshold=args.th_list)


# The remaining part of the code is not obligatory
# It is added to probide insight for the usage
# This part visualizes the Task 1 and Task 2 Scores for different FW ratios
'''
ratios = np.arange(start=0.05, stop=1.05, step=0.05)
#results are stored as r, en_res, de_res, sw_res, lt_res, avg_res  
FW_binary_cd = [ratios, [], [], [], [], []]
FW_spearman_cd = [ratios, [], [], [], [], []]
for r in ratios:
    bin_cd_R, spear_cd_R,_,_ = StoreDistances(embedding='Normal', sharedVocab = 'frequent', thresholdMethod ='Gamma', ratio=r)
    for i in range(1,6):
        FW_binary[i].append(bin_R[i-1])
        FW_spearman[i].append(spear_R[i-1])


ratios = 100*np.around(ratios,decimals=2)
plt.figure()
plt.plot(ratios, FW_binary[1], label="English")
plt.plot(ratios, FW_binary[2], label="German")
plt.plot(ratios, FW_binary[3], label="Swedish")
plt.plot(ratios, FW_binary[4], label="Latin")
plt.plot(ratios, FW_binary[5], label="Average",linewidth=3)
plt.legend()
plt.xlabel("FW Ratio percentage", fontsize = 10)
plt.ylabel("Binary Classification Accuracy", fontsize = 10)
plt.savefig("RevisionFiles/binary.png", format = "png", dpi = 150, bbox_inches = "tight")


plt.figure()
plt.plot(ratios, FW_spearman[1], label="English")
plt.plot(ratios, FW_spearman[2], label="German")
plt.plot(ratios, FW_spearman[3], label="Swedish")
plt.plot(ratios, FW_spearman[4], label="Latin")
plt.plot(ratios, FW_spearman[5], label="Average",linewidth=3)
plt.legend()
plt.xlabel("FW Ratio percentage", fontsize = 10)
plt.ylabel("Spearman Correleation", fontsize = 10)
plt.savefig("RevisionFiles/spearman.png", format = "png", dpi = 150, bbox_inches = "tight")

'''

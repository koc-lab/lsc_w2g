from word2gm_loader import Word2GM
from quantitative_eval import *
import matplotlib.pyplot as plt
import pandas as pd

# Load Unimodal
"""
model_dir_c2_uni = 'models/English/c2Unimodal/'
model_dir_c1_uni = "models/English/c1Unimodal/"
w2gm_2s_c2_uni = Word2GM(model_dir_c2_uni)
w2gm_2s_c1_uni = Word2GM(model_dir_c1_uni)
"""

# Load Bimodal

model_dir_c2_uni = 'models/English/c2Bimodal/'
model_dir_c1_uni = "models/English/c1Bimodal/"
w2gm_2s_c2_uni = Word2GM(model_dir_c2_uni)
w2gm_2s_c1_uni = Word2GM(model_dir_c1_uni)



# Find Words that are common in both corpora
selected_words = ['attack_nn', 'bag_nn', 'ball_nn', 'bit_nn', 'chairman_nn', 'circle_vb', 'contemplation_nn', 'donkey_nn', 'edge_nn', 'face_nn', 'fiction_nn', 'gas_nn', 'graft_nn', 'head_nn', 'land_nn', 'lane_nn', 'lass_nn', 'multitude_nn', 'ounce_nn', 'part_nn', 'pin_vb', 'plane_nn', 'player_nn', 'prop_nn', 'quilt_nn', 'rag_nn', 'record_nn', 'relationship_nn', 'risk_nn', 'savage_nn', 'stab_nn', 'stroke_vb', 'thump_nn', 'tip_vb', 'tree_nn', 'twist_nn', 'word_nn']

results = ['1', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0', '1', '1', '1', '0', '1', '0', '0', '0', '0', '1', '1', '1', '0', '1', '1', '0', '0', '0', '1', '0', '1', '1', '0', '0', '0']


c1_mix_0 = w2gm_2s_c1_bi.getMixes(selected_words, 0)
c2_mix_0 = w2gm_2s_c2_bi.getMixes(selected_words, 0)
c1_mix_1 = w2gm_2s_c1_bi.getMixes(selected_words, 1)
c2_mix_1 = w2gm_2s_c2_bi.getMixes(selected_words, 1)
mix_diff_00 = []
mix_diff_01 = []
mix_diff_10 = []
mix_diff_11 = []
for i in range(0,len(c2_mix_0)):
    mix_diff_00.append(c2_mix_0[i]-c1_mix_0[i])
    mix_diff_01.append(c2_mix_0[i]-c1_mix_1[i])
    mix_diff_11.append(c2_mix_1[i]-c1_mix_1[i])
    mix_diff_10.append(c2_mix_1[i]-c1_mix_0[i])  

mix_diff = mix_diff_00 + mix_diff_01 + mix_diff_10 + mix_diff_11
mix_diff = [abs(mix) for mix in mix_diff]
minVal = min(mix_diff)
print(minVal)

mix_diff_00_new = [mix/minVal for mix in mix_diff_00]
mix_diff_01_new = [mix/minVal for mix in mix_diff_01]
mix_diff_10_new = [mix/minVal for mix in mix_diff_10]
mix_diff_11_new = [mix/minVal for mix in mix_diff_11]

df_common = pd.DataFrame({'words': selected_words, 'Binary Results': results, 'Variance Difference_00': mix_diff_00, 'Variance Difference_11': mix_diff_11, 'Variance Difference_10': mix_diff_10, 'Variance Difference_01': mix_diff_01})
df_common = df_common.sort_values(by=['Binary Results'],ascending=False)
df_common.to_excel("bimodalSelectedWords.xlsx")

df_common = pd.DataFrame({'words': selected_words, 'Binary Results': results, 'Variance Difference_00': mix_diff_00_new, 'Variance Difference_11': mix_diff_11_new, 'Variance Difference_10': mix_diff_10_new, 'Variance Difference_01': mix_diff_01_new})
df_common = df_common.sort_values(by=['Binary Results'],ascending=False)
df_common.to_excel("bimodalSelectedWordsWeighted.xlsx")  


c1  = open("models/English/c1Bimodal/vocab.txt")
c2  = open("models/English/c2Bimodal/vocab.txt")
c1_words =set()
c2_words =set()
for line in c1:
    line = line.split()
    c1_words.add(line[0])

for line in c2:
    line = line.split()
    c2_words.add(line[0])
 
common_words = c2_words.intersection(c1_words)
common_words = list(common_words)
selected_words = common_words
c1_mix_0 = w2gm_2s_c1_bi.getMixes(selected_words, 0)
c2_mix_0 = w2gm_2s_c2_bi.getMixes(selected_words, 0)
c1_mix_1 = w2gm_2s_c1_bi.getMixes(selected_words, 1)
c2_mix_1 = w2gm_2s_c2_bi.getMixes(selected_words, 1)
mix_diff_00 = []
mix_diff_01 = []
mix_diff_10 = []
mix_diff_11 = []
for i in range(0,len(c2_mix_0)):
    mix_diff_00.append(c2_mix_0[i]-c1_mix_0[i])
    mix_diff_01.append(c2_mix_0[i]-c1_mix_1[i])
    mix_diff_11.append(c2_mix_1[i]-c1_mix_1[i])
    mix_diff_10.append(c2_mix_1[i]-c1_mix_0[i])  

mix_diff = mix_diff_00 + mix_diff_01 + mix_diff_10 + mix_diff_11
mix_diff = [abs(mix) for mix in mix_diff]
minVal = min(mix_diff)
print(minVal)

mix_diff_00_new = [mix/minVal for mix in mix_diff_00]
mix_diff_01_new = [mix/minVal for mix in mix_diff_01]
mix_diff_10_new = [mix/minVal for mix in mix_diff_10]
mix_diff_11_new = [mix/minVal for mix in mix_diff_11]


df_common = pd.DataFrame({'words': selected_words,  'Variance Difference_00': mix_diff_00, 'Variance Difference_11': mix_diff_11, 'Variance Difference_10': mix_diff_10, 'Variance Difference_01': mix_diff_01})
df_common.to_excel("bimodalSCommonWords.xlsx")

df_common = pd.DataFrame({'words': selected_words, 'Variance Difference_00': mix_diff_00_new, 'Variance Difference_11': mix_diff_11_new, 'Variance Difference_10': mix_diff_10_new, 'Variance Difference_01': mix_diff_01_new})
df_common.to_excel("bimodalCommonWordsWeighted.xlsx") 
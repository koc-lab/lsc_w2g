
# This code performs local and global neighbourhood change for an example_word


from word2gm_loader import Word2GM
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from quantitative_eval import *
import numpy as np
import scipy
import plotly.graph_objects as go
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

# Insert here the directories of w2gm models which you want to compare
c1_dir = 'first_w2gm_model_directory'
c2_dir = 'second_w2gm_model_directory'

# Creating w2gm objects
w2gm = Word2GM(c1_dir)
w2gm2 = Word2GM(c2_dir)

# Visualizing w2gm models
w2gm.visualize_embeddings()
w2gm2.visualize_embeddings()

# This method returns the mu value of a word in first_w2gm_model


def get_mu_unimodal_w2gm(word):
    id = w2gm.get_idx(word)
    cl = 0
    return w2gm.mus_n[id*w2gm.num_mixtures + cl]

# This method returns the mu value of a word in second_w2gm_model


def get_mu_unimodal_w2gm2(word):
    id = w2gm2.get_idx(word)
    cl = 0
    return w2gm2.mus_n[id*w2gm2.num_mixtures + cl]

# This method returns the words which are the intersection of the words in first_w2gm_model and second_w2gm_model


def find_common_words(lang='en'):
    list_temp = set(w2gm.id2word) & set(w2gm2.id2word)
    common_words = list_temp
    # removing stop words
    y = stopwords.words('english')
    r = [item for item in common_words if item not in y and item.isalpha()]
    res = sorted(r, key=lambda k: w2gm2.word2id[k])
    return res

# This method returns the mean vectors of the words in common_words for both w2gm model


def intersection_alignment(lang='en'):
    if(lang == 'lt'):
        common_words = find_common_words(lang)
    else:
        common_words = find_common_words(lang)
    newvectors1 = np.empty(
        (len(common_words), get_mu_unimodal_w2gm(common_words[0]).shape[0]))
    newvectors2 = np.empty(
        (len(common_words), get_mu_unimodal_w2gm2(common_words[0]).shape[0]))
    for i in range(len(common_words)):
        newvectors1[i] = get_mu_unimodal_w2gm(common_words[i])
        newvectors2[i] = get_mu_unimodal_w2gm2(common_words[i])
    return (newvectors1, newvectors2, common_words)

# This method performs procrustes alignment, hence making both embedding spaces comparable


def procrustes_alignment(lang='en'):
    global w2gm
    global w2gm2
    (new_vectors1, new_vectors2, common_words) = intersection_alignment(lang)
    r, scale = scipy.linalg.orthogonal_procrustes(new_vectors2, new_vectors1)
    w2gm2.mus_n = np.dot(w2gm2.mus_n, r)
    return common_words


# Getting common_words
common_words = procrustes_alignment()


# Local neighbourhood change
word = 'example_word'

(words1, dist_val1, var_val1) = w2gm.semChangeValues(example_word)
(words2, dist_val2, var_val2) = w2gm2.semChangeValues(example_word)

# Getting information of n most nearest words of example_word for boths w2gm models
words = np.concatenate([words1, words2])
dist_val = np.concatenate([dist_val1, dist_val2])
var_val = np.concatenate([var_val1, var_val2])

print(words)
print(dist_val)
print(var_val)

# Visualizing the local neighbourhood change
p1 = plt.scatter(x=dist_val1, y=var_val1, c='b', linewidth=1, marker='x')
p2 = plt.scatter(x=dist_val2, y=var_val2, c='r', linewidth=1)
i = 0
for x, y in zip(dist_val, var_val):
    plt.annotate(words[i], (x, y), textcoords='offset pixels', xytext=(-10, 6))
    i = i + 1

plt.grid()
plt.xlim([0.70, 1.10])
plt.ylim([-3, -2.75])
plt.title('Local', fontsize=12, fontweight='bold')
plt.xlabel('similarity', fontsize=10, fontweight='bold')
plt.ylabel('log-variance', fontsize=10, fontweight='bold')
plt.legend((p1, p2), ('year_of_first_w2gm_model', 'year_of_second_w2gm_model'))
plt.yticks(weight='bold')
plt.xticks(weight='bold')
plt.savefig('local.png')
plt.close()


# Global neighbourhood change
# words[] consist of 12 words. The first 6 words of words[] consist of the words which are in the nearest neighbourhood of the example_word in first_w2gm_model, the second 6 words of words[] consist of the words which are in the nearest neighbourhood of the example_word in second_w2gm_model

# Reducing to 2 dimensions
pca = PCA(n_components=2)

# Creating 12 50-dimensional vectors for pca
vectors = np.zeros((12, 50))

# Getting the mu values for the 12 words in words[]
for i in range(6):
    vectors[i, :] = get_mu_unimodal_w2gm(words[i][:-2])
for j in range(6, 12):
    vectors[j, :] = get_mu_unimodal_w2gm2(words[j][:-2])

# print(vectors)
principalComponents = pca.fit_transform(vectors)
# print(pca.explained_variance_ratio_)
print(principalComponents)

x1 = np.zeros(6)
x2 = np.zeros(6)
y1 = np.zeros(6)
y2 = np.zeros(6)

for i in range(0, 6):
    x1[i] = principalComponents[i, 0]
    y1[i] = principalComponents[i, 1]

for j in range(6, 12):
    x2[j-6] = (principalComponents[j])[0]
    y2[j-6] = (principalComponents[j])[1]

print(x1)
print(y1)
print(x2)
print(y2)

# Visalizing the PCA
p3 = plt.scatter(x=x1, y=y1, c='b', linewidth=1.5, marker='x')
p4 = plt.scatter(x=x2, y=y2, c='r', linewidth=1.5)
t = 0
for x, y in zip(np.append(x1, x2), np.append(y1, y2)):
    plt.annotate(words[t], (x, y), textcoords='offset pixels', xytext=(-20, 4))
    t = t + 1

# Visualizing global neighbourhood change
plt.grid()
plt.xlim([-0.5, 0.6])
plt.ylim([-0.5, 0.6])
plt.title('Global', fontsize=12, fontweight='bold')
plt.xlabel('Principal Component 1, Variance: %' +
           (str(pca.explained_variance_ratio_[0] * 100)[:4]), fontsize=10, fontweight='bold')
plt.ylabel('Principal Component 2, Variance: %' +
           (str(pca.explained_variance_ratio_[1] * 100)[:4]), fontsize=10, fontweight='bold')
plt.legend((p3, p4), (), ('year_of_first_w2gm_model',
           'year_of_second_w2gm_model'))
plt.yticks(weight='bold')
plt.xticks(weight='bold')
plt.savefig('global.png')
plt.close()

import codecs
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer


def preprocess_corpus(lang, corp):
    path = "data/semEval/"
    print("Started Language {} with corpus {}".format(lang, corp))
    file1 = path+lang+"/"+corp
    file2 = path+lang+"/clean_"+corp
    tk = WhitespaceTokenizer()
    if(lang == "en-semeval"):
        stop_words = set(stopwords.words('english'))
    elif(lang == "de-semeval"):
        stop_words = set(stopwords.words('german'))
    elif(lang == "sw-semeval"):
        stop_words = set(stopwords.words('swedish'))
    else:
        stop_words = set(["ab", "ac", "ad", "adhic", "aliqui", "aliquis", "an", "ante", "apud", "at", "atque", "aut", "autem", "cum", "cur", "de", "deinde", "dum", "ego", "enim", "ergo", "es", "est", "et", "etiam", "etsi", "ex", "fio", "haud", "hic", "iam", "idem", "igitur", "ille", "in", "infra", "inter", "interim", "ipse", "is", "ita", "magis", "modo", "mox", "nam", "ne",
                         "nec", "necque", "neque", "nisi", "non", "nos", "o", "ob", "per", "possum", "post", "pro", "quae", "quam", "quare", "qui", "quia", "quicumque", "quidem", "quilibet", "quis", "quisnam", "quisquam", "quisque", "quisquis", "quo", "quoniam", "sed", "si", "sic", "sive", "sub", "sui", "sum", "super", "suus", "tam", "tamen", "trans", "tu", "tum", "ubi", "uel", "uero"])
    new_words = ["/z/", "**", "//", "" "--"]
    stop_words = stop_words.union(new_words)
    with codecs.open(file1, 'r', 'utf-8') as infile:
        lines = infile.readlines()
        with codecs.open(file2, 'w', 'utf-8') as outfile:
            for line in lines:
                word_tokens = tk.tokenize(line)
                filtered_sentence = []
                for w in word_tokens:
                    if(w.isdigit()):
                        filtered_sentence.append("<NUM>")
                    elif(w not in stop_words):
                        filtered_sentence.append(w)
                s = " ".join(filtered_sentence)
                s = s+'\n'
                outfile.write(s)
    print("Finished Language {} with corpus {}".format(lang, corp))


if __name__ == '__main__':
    processes = []
    languages = ["en-semeval", "de-semeval", "ln-semeval", "sw-semeval"]
    text = ["c1.txt", "c2.txt"]
    for i in range(0, 4):
        p = multiprocessing.Process(
            target=preprocess_corpus, args=(languages[i], text[0]))
        processes.append(p)
        p.start()
        p = multiprocessing.Process(
            target=preprocess_corpus, args=(languages[i], text[1]))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

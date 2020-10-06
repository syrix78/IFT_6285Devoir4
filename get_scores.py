import os

from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser


if __name__ == '__main__':

    print("Hello World!")

    sentence_stream = "./samples/"

    bigram_reloaded = Phraser.load("./bigram_model.pkl")
    bigram_scores = bigram_reloaded.phrasegrams
    trigram_reloaded = Phraser.load(("./trigram_model.pkl"))
    trigram_scores = trigram_reloaded.phrasegrams

    #https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    sorted_scores = {k: v for k, v in sorted(bigram_scores.items(), key=lambda item: item[1])}

    print(sorted_scores)



import os

from gensim.test.utils import datapath
from gensim.utils import simple_preprocess, simple_tokenize
from gensim.models.word2vec import Text8Corpus, PathLineSentences
from gensim.models.phrases import Phrases, Phraser

if __name__ == '__main__':

    print("Hello World!")

    path = "./test_folder/"

    #Olivier Nadeau sur le forum Slack du Cours IFT6285
    corpus = PathLineSentences(path)

    print("Training Bigram")
    #https://medium.com/@manjunathhiremath.mh/identifying-bigrams-trigrams-and-four-grams-using-word2vec-dea346130eb
    bigram = Phrases(corpus, min_count=1)
    print("Training Trigram")
    trigram = Phrases(bigram[corpus], min_count=1)

    print("Bigram Phraser")
    bigram_phraser = Phraser(bigram)
    bigram_phraser.save("bigram_model.pkl")

    print("Trigram Phraser")
    trigram_phraser = Phraser(trigram)
    trigram_phraser.save("trigram_model.pkl")




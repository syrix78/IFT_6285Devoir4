import os

from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus, PathLineSentences
from gensim.models.phrases import Phrases, Phraser


if __name__ == '__main__':

    print("Hello World!")

    bigram_reloaded = Phraser.load("./bigram_model.pkl")
    bigram_scores = bigram_reloaded.phrasegrams
    trigram_reloaded = Phraser.load("./trigram_model.pkl")
    trigram_scores = trigram_reloaded.phrasegrams


    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    sorted_scores = {k: v for k, v in sorted(bigram_scores.items(), key=lambda item: item[1], reverse=True)}
    sorted_bigrams = {k: v for k, v in sorted_scores.items() if str(k).count("_") == 0}

    """
    Explication: Les n-grams sont séparés par des "_" lorsque l'on construit le modele dans Phrase.
    Puisque Phrasers.phrasegrams retourne des bigrams, il suffit de compter le nombre de "_" dans les clés de phrasegrams
    puis tous les bigrams avec un seul "_" dedans sont des trigrams.
    
    D'après ce que j'ai lu sur le net, Gensim est vraiment fait pour les bigrams et utilise les propritétés des bigrams 
    pour générer des trigrams avec les bigrams. Gensim genere en meme temps des 4 grams et 5 grams sans que je comprene
    pourquoi (Les gens sur internet ne comprenent d'ailleurs pas non plus)
    """
    sorted_scores = {k: v for k, v in sorted(trigram_scores.items(), key=lambda item: item[1], reverse=True)}
    sorted_trigrams = {k: v for k, v in sorted_scores.items() if str(k).count("_") == 1}



    print(sorted_bigrams)
    print(sorted_trigrams)



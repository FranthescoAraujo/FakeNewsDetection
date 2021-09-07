from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from TermFrequencyInverseDocumentFrequency import TermFrequencyInverseDocumentFrequency
import os

from gensim import corpora

CORPUS_PATH = "../Corpus/"

files = os.listdir(CORPUS_PATH + "Fake/")

listNews = []

for file in files:
    f = open(CORPUS_PATH + "Fake/" + file, "r")
    listNews.append(f.read())

newlistNews = PreProcessing.removeAccentuation(listNews)
newlistNews = PreProcessing.removeSpecialCharacters(newlistNews)
newlistNews = PreProcessing.removeNumerals(newlistNews)
newlistNews = PreProcessing.toLowerCase(newlistNews)
#listNews = PreProcessing.toSplit(listNews)

print(TermFrequencyInverseDocumentFrequency.createVectors(newlistNews))
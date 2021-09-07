from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
import tensorflow as tf
import os

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

teste = DocumentRepresentationDoc2Vec(newlistNews)
print(teste.documents)
print(teste.paragraphVectorDistributedMemory())

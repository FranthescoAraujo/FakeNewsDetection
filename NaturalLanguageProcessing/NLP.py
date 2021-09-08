from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from TermFrequencyInverseDocumentFrequency import TermFrequencyInverseDocumentFrequency
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
import os

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

doc2vec = DocumentRepresentationDoc2Vec(newlistNews)
listDoc2vecDM = doc2vec.paragraphVectorDistributedMemory()
listDoc2vecDBOW = doc2vec.paragraphVectorDistributedBagOfWords()
listDoc2vecConcat = doc2vec.concatBothParagraphVectors()

word2vec = DocumentRepresentationWord2Vec(newlistNews)
listWord2VecSkipGram = word2vec.skipGramDocumentRepresentation()
listWord2VecCBOW = word2vec.continuousBagOfWordsDocumentRepresentation()

tfidf = TermFrequencyInverseDocumentFrequency()
listTfIdf = tfidf.createVectors(newlistNews)

# print("########################################### DOC2VEC - DM ###########################################")
# print(len(listDoc2vecDM[0]))
# print(listDoc2vecDM)
# print("########################################### DOC2VEC - DBOW ###########################################")
# print(len(listDoc2vecDBOW[0]))
# print(listDoc2vecDBOW)
# print("########################################### DOC2VEC - Concat ###########################################")
# print(len(listDoc2vecConcat[0]))
# print(listDoc2vecConcat)
# print("########################################### WORD2VEC - SkipGram ###########################################")
# print(len(listWord2VecSkipGram[0]))
# print(listWord2VecSkipGram)
# print("########################################### WORD2VEC - CBOW ###########################################")
# print(len(listWord2VecCBOW[0]))
# print(listWord2VecCBOW)
# print("########################################### TF - IDF ###########################################")
# print(len(listTfIdf[0]))
# print(listTfIdf)






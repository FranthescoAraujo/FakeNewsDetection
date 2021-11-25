import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from TermFrequencyInverseDocumentFrequency import TermFrequencyInverseDocumentFrequency
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
from Classifiers import Classifiers

datasetPortugues = True
removeStopWords = True
portugues = True

tic = time.time()
CORPUS_PATH = "../Corpus/Ingles/"
if datasetPortugues:
    CORPUS_PATH = "../Corpus/Portugues/"

folders = os.listdir(CORPUS_PATH)
listNews = []
listLabels = []
listTexts = []
for folder in folders:
    for file in os.listdir(CORPUS_PATH + folder):
        f = open(CORPUS_PATH + folder + "/" + file, "r")
        listTexts.append(CORPUS_PATH + folder + "/" + file)
        listNews.append(f.read())
        if folder == "Fake":
            listLabels.append(1)
            continue
        listLabels.append(0)
toc = time.time() - tic
print("Etapa 01 - Carregando Dataset - " + str(round(toc,2)) + " segundos")

tic = time.time()
newlistNews = PreProcessing.removeAccentuation(listNews)
newlistNews = PreProcessing.removeSpecialCharacters(newlistNews)
newlistNews = PreProcessing.removeNumerals(newlistNews)
newlistNews = PreProcessing.toLowerCase(newlistNews)
if removeStopWords:
    newlistNews = PreProcessing.removeStopWords(newlistNews, portugues)

toc = time.time() - tic
print("Etapa 02 - Pré-processamento - " + str(round(toc,2)) + " segundos")

tic = time.time()
# REPRESENTAÇÃO DOC2VEC
doc2vec = DocumentRepresentationDoc2Vec(newlistNews)
listVectors = doc2vec.paragraphVectorDistributedMemory(vector_size=300)
# listVectors = doc2vec.paragraphVectorDistributedBagOfWords()
# listVectors = doc2vec.concatBothParagraphVectors()
# REPRESENTAÇÃO WORD2VEC
# word2vec = DocumentRepresentationWord2Vec(newlistNews)
# listVectors = word2vec.skipGramMatrixDocumentRepresentation(vector_size=100, matrix_size=300)
# listVectors = word2vec.skipGramDocumentRepresentation()
# listVectors = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=1)
# REPRESENTAÇÃO TF-IDF
# tfidf = TermFrequencyInverseDocumentFrequency()
# tfidf.countWords(newlistNews)
# listTfIdf = tfidf.createVectors(newlistNews)
# tfidf.wordInDocument(newlistNews)
toc = time.time() - tic
print("Etapa 03 - Processamento de Linguagem Natural - " + str(round(toc,2)) + " segundos")

tic = time.time()
classificador = Classifiers(listVectors, listLabels)
classificador.neuralNetwork(input_size=300)
toc = time.time() - tic
print("Etapa 04 - Treinamento da Rede - " + str(round(toc,2)) + " segundos")
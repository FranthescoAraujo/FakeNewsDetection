from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from TermFrequencyInverseDocumentFrequency import TermFrequencyInverseDocumentFrequency
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
from Classifiers import Classifiers
import time
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

tic = time.time()
CORPUS_PATH = "../Corpus/Portugues/"
#CORPUS_PATH = "../Corpus/Ingles/"
folders = os.listdir(CORPUS_PATH)
listNews = []
listLabel = []
listText = []
for folder in folders:
    for file in os.listdir(CORPUS_PATH + folder):
        f = open(CORPUS_PATH + folder + "/" + file, "r")
        listText.append(CORPUS_PATH + folder + "/" + file)
        listNews.append(f.read())
        if folder == "Fake":
            listLabel.append(1)
            continue
        listLabel.append(0)
toc = time.time() - tic
print("Etapa 01 - Carregando Dataset - " + str(toc) + " segundos")

tic = time.time()
newlistNews = PreProcessing.removeAccentuation(listNews)
newlistNews = PreProcessing.removeSpecialCharacters(newlistNews)
newlistNews = PreProcessing.removeNumerals(newlistNews)
newlistNews = PreProcessing.toLowerCase(newlistNews)
toc = time.time() - tic
print("Etapa 02 - Pré-processamento - " + str(toc) + " segundos")

tic = time.time()
# REPRESENTAÇÃO DOC2VEC
# doc2vec = DocumentRepresentationDoc2Vec(newlistNews)
# listDoc2vecDM = doc2vec.paragraphVectorDistributedMemory(vector_size=300)
# listDoc2vecDBOW = doc2vec.paragraphVectorDistributedBagOfWords()
# listDoc2vecConcat = doc2vec.concatBothParagraphVectors()
# REPRESENTAÇÃO WORD2VEC
word2vec = DocumentRepresentationWord2Vec(newlistNews)
listMatrixWord2VecSkipGram = word2vec.skipGramMatrixDocumentRepresentation()
# listWord2VecSkipGram = word2vec.skipGramDocumentRepresentation()
# listWord2VecCBOW = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=1)
# REPRESENTAÇÃO TF-IDF
# tfidf = TermFrequencyInverseDocumentFrequency()
# tfidf.countWords(newlistNews)
# listTfIdf = tfidf.createVectors(newlistNews)
# tfidf.wordInDocument(newlistNews)
toc = time.time() - tic
print("Etapa 03 - Processamento de Linguagem Natural - " + str(toc) + " segundos")

tic = time.time()
npList = []
for document in listMatrixWord2VecSkipGram:
    npList.append(np.array(document))
npListLabel = []
for label in listLabel:
    npListLabel.append(np.array(label))
npList = np.array(npList)
npListLabel = np.array(npListLabel)
toc = time.time() - tic
print("Etapa 04 - Conversão Numpy Array - " + str(toc) + " segundos")

tic = time.time()
X_train, X_test, y_train, y_test = train_test_split(npList, npListLabel, test_size=0.2, random_state=42)
toc = time.time() - tic
print("Etapa 05 - Separação do dataset em treino e teste - " + str(toc) + " segundos")

tic = time.time()
# pca = PCA(n_components=100)
# npList = pca.fit_transform(npList)
# npList = np.expand_dims(npList, axis=2)
toc = time.time() - tic
print("Etapa 06 - Redução de Dimensionalidade Utilizando PCA - " + str(toc) + " segundos")

tic = time.time()
classificador = Classifiers(X_train, y_train, X_test, y_test)
y_pred = classificador.longShortTermMemory()
toc = time.time() - tic
print("Etapa 07 - Treinamento da Rede - " + str(toc) + " segundos")

tic = time.time()
plot_confusion_matrix(y_test, y_pred)
plt.savefig('word2vec - LSTM.png')
toc = time.time() - tic
print("Etapa 08 - Plotando Matrix de Confusão - " + str(toc) + " segundos")
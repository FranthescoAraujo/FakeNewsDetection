from gensim import models
from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from TermFrequencyInverseDocumentFrequency import TermFrequencyInverseDocumentFrequency
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
import time
import os

import tensorflow.keras as keras
import numpy as np
from sklearn.decomposition import PCA

tic = time.time()

CORPUS_PATH = "../Corpus/"
#CORPUS_PATH = "../Teste/"

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
# listDoc2vecDM = doc2vec.paragraphVectorDistributedMemory()
# listDoc2vecDBOW = doc2vec.paragraphVectorDistributedBagOfWords()
# listDoc2vecConcat = doc2vec.concatBothParagraphVectors()

# REPRESENTAÇÃO WORD2VEC
# word2vec = DocumentRepresentationWord2Vec(newlistNews)
# listWord2VecSkipGram = word2vec.skipGramDocumentRepresentation(vector_size=200)
# listWord2VecCBOW = word2vec.continuousBagOfWordsDocumentRepresentation()

# REPRESENTAÇÃO TF-IDF
# tfidf = TermFrequencyInverseDocumentFrequency()
# tfidf.countWords(newlistNews)
# listTfIdf = tfidf.createVectors(newlistNews)
# tfidf.wordInDocument(newlistNews)

toc = time.time() - tic

print("Etapa 03 - Processamento de Linguagem Natural - " + str(toc) + " segundos")

tic = time.time()

npList = []
for document in listTfIdf:
    npList.append(np.array(document))
npListLabel = []
for label in listLabel:
    npListLabel.append(np.array(label))

npList = np.array(npList)
npListLabel = np.array(npListLabel)

print(len(npList[0]))

toc = time.time() - tic

print("Etapa 04 - Conversão Numpy Array - " + str(toc) + " segundos")

tic = time.time()

# pca = PCA(n_components=100)
# npList = pca.fit_transform(npList)
# npList = np.expand_dims(npList, axis=2)

toc = time.time() - tic

print("Etapa 05 - Redução de Dimensionalidade Utilizando PCA - " + str(toc) + " segundos")

tic = time.time()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100,1)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(npList, npListLabel, epochs=10)

toc = time.time() - tic

print("Etapa 06 - Treinamento da Rede - " + str(toc) + " segundos")
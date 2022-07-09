import time
import json
import os
import numpy as np
from PreProcessing import PreProcessing
from datetime import datetime
from sklearn.model_selection import train_test_split

def tempoAgora():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def salvarJson(path, dataset, removeStopWords, list, labels, name):
    data = {"dataset":list, "labels":labels}
    if not os.path.exists(path + dataset + "/RemoveStopWords-" + str(removeStopWords)):
        os.makedirs(path + dataset + "/RemoveStopWords-" + str(removeStopWords))
    f = open(path + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + name + ".json", "w")
    json.dump(data, f)
    f.close()

def preProcessingData(PATH_JSON, dataset, removeStopWords, listNews, listLabels, name):
    listNews = PreProcessing.removeAccentuation(listNews)
    listNews = PreProcessing.removeSpecialCharacters(listNews)
    listNews = PreProcessing.removeNumerals(listNews)
    listNews = PreProcessing.toLowerCase(listNews)
    if removeStopWords:
        listNews = PreProcessing.removeStopWords(listNews, dataset)
    # listNews = PreProcessing.convertWordsToStemming(listNews, dataset)
    listNews, listLabels = PreProcessing.removeDocumentsWithFewWords(listNews, listLabels)
    # newlistNews = PreProcessing.removeDocumentsWithManyWords(newlistNews, dataset)
    # calcularNumeroPalavras(newlistNews, listLabels, dataset, removeStopWords)
    salvarJson(PATH_JSON, dataset, removeStopWords, listNews, listLabels, name)
    del listNews, listLabels

def calcularNumeroPalavras(documents, listLabels, dataset, removeStopWords):
    mediaNumeroPalavrasTrue = 0
    minNumeroPalavrasTrue = 10000
    maxNumeroPalavrasTrue = 0
    totalWordsTrue = 0
    mediaNumeroPalavrasFalse = 0
    minNumeroPalavrasFalse = 10000
    maxNumeroPalavrasFalse = 0
    totalWordsFalse = 0
    for index, document in enumerate(documents):
        numWordsTrue = 0
        numWordsFalse = 0
        if (listLabels[index] == 0):
            totalWordsTrue += 1
            for word in document.split():
                numWordsTrue += 1
            if (minNumeroPalavrasTrue > numWordsTrue):
                minNumeroPalavrasTrue = numWordsTrue
            if (maxNumeroPalavrasTrue < numWordsTrue):
                maxNumeroPalavrasTrue = numWordsTrue
        else:
            totalWordsFalse += 1
            for word in document.split():
                numWordsFalse += 1
            if (minNumeroPalavrasFalse > numWordsFalse):
                minNumeroPalavrasFalse = numWordsFalse
            if (maxNumeroPalavrasFalse < numWordsFalse):
                maxNumeroPalavrasFalse = numWordsFalse
        mediaNumeroPalavrasTrue += numWordsTrue
        mediaNumeroPalavrasFalse += numWordsFalse
    mediaNumeroPalavrasTrue = mediaNumeroPalavrasTrue/totalWordsTrue
    mediaNumeroPalavrasFalse = mediaNumeroPalavrasFalse/totalWordsFalse

    print("TotalDocumentosTrue = " + str(totalWordsTrue))
    print("Dataset = " + dataset + " RemoveStopWords = " + str(removeStopWords) + " mediaNumeroPalavrasTrue = " + str(mediaNumeroPalavrasTrue))
    print("Dataset = " + dataset + " RemoveStopWords = " + str(removeStopWords) + " minNumeroPalavrasTrue = " + str(minNumeroPalavrasTrue))
    print("Dataset = " + dataset + " RemoveStopWords = " + str(removeStopWords) + " maxNumeroPalavrasTrue = " + str(maxNumeroPalavrasTrue))
    print("TotalDocumentosFalse = " + str(totalWordsFalse))
    print("Dataset = " + dataset + " RemoveStopWords = " + str(removeStopWords) + " mediaNumeroPalavrasFalse = " + str(mediaNumeroPalavrasFalse))
    print("Dataset = " + dataset + " RemoveStopWords = " + str(removeStopWords) + " minNumeroPalavrasFalse = " + str(minNumeroPalavrasFalse))
    print("Dataset = " + dataset + " RemoveStopWords = " + str(removeStopWords) + " maxNumeroPalavrasFalse = " + str(maxNumeroPalavrasFalse))

def trainTestSplit(listNews, listLabels, test_size=0.3, random_state=42):
    listNewsIds = list(range(len(listNews)))
    listNewsTrainIds, listNewsTestIds, yTrain, yTest = train_test_split(np.array(listNewsIds), np.array(listLabels), test_size=0.3, random_state=42)
    returnListNewsTrain = []
    returnListNewsTest = []
    for id in listNewsTrainIds:
        returnListNewsTrain.append(listNews[id])
    for id in listNewsTestIds:
        returnListNewsTest.append(listNews[id])
    return returnListNewsTrain, returnListNewsTest, yTrain.tolist(), yTest.tolist()

PATH_JSON = "../Json/"
PATH_JSON_TEST = "../JsonTest/"
dataSetCsv = ["Inglês"]
removeStopWordsCsv = [True, False]
# dataSetCsv = ["Português"]
# removeStopWordsCsv = [True, False]

for dataset in dataSetCsv:
    for removeStopWords in removeStopWordsCsv:
        tic = time.time()
        CORPUS_PATH = "../Corpus/Inglês/"
        if dataset == "Português":
            CORPUS_PATH = "../Corpus/Português/"
        listNews = []
        listLabels = []
        for folder in os.listdir(CORPUS_PATH):
            for file in os.listdir(CORPUS_PATH + folder):
                f = open(CORPUS_PATH + folder + "/" + file, "r")
                listNews.append(f.read())
                if folder == "Fake":
                    listLabels.append(1)
                    continue
                listLabels.append(0)
        f.close()
        toc = time.time() - tic
        print(tempoAgora() + " - Carregando dataset " + dataset + " - " + str(round(toc,2)) + " segundos")

        tic = time.time()
        listNewsTrain, listNewsTest, yTrain, yTest = trainTestSplit(listNews, listLabels)
        del listNews, listLabels
        salvarJson(PATH_JSON_TEST, dataset, removeStopWords, listNewsTest, yTest, "datasetTest")
        preProcessingData(PATH_JSON, dataset, removeStopWords, listNewsTrain, yTrain, "datasetTrain")
        preProcessingData(PATH_JSON, dataset, removeStopWords, listNewsTest, yTest, "datasetTest")
        toc = time.time() - tic
        print(" " + tempoAgora() + " - Pré-processamento - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos")
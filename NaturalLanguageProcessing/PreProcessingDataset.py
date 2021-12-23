import time
import json
import os
from PreProcessing import PreProcessing
from datetime import datetime

def tempoAgora():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def salvarJson(path, dataset, removeStopWords, list, labels):
    data = {"dataset":list, "labels":labels}
    f = open(path + dataset + "/stopWords-" + str(removeStopWords) + "/dataset.json", "w")
    json.dump(data, f)
    f.close()

PATH_JSON = "../Json/"
dataSetCsv = ["português", "inglês"]
removeStopWordsCsv = [True, False]

for dataset in dataSetCsv:
    for removeStopWords in removeStopWordsCsv:
        tic = time.time()
        CORPUS_PATH = "../Corpus/inglês/"
        if dataset == "português":
            CORPUS_PATH = "../Corpus/português/"
        folders = os.listdir(CORPUS_PATH)
        listNews = []
        listLabels = []
        for folder in folders:
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
        newlistNews = PreProcessing.removeAccentuation(listNews)
        newlistNews = PreProcessing.removeSpecialCharacters(newlistNews)
        newlistNews = PreProcessing.removeNumerals(newlistNews)
        newlistNews = PreProcessing.toLowerCase(newlistNews)
        if removeStopWords:
            newlistNews = PreProcessing.removeStopWords(newlistNews, dataset)
        newlistNews, listLabels = PreProcessing.removeDocumentsWithFewWords(newlistNews, listLabels)
        salvarJson(PATH_JSON, dataset, removeStopWords, newlistNews, listLabels)
        del newlistNews, listLabels
        toc = time.time() - tic
        print(" " + tempoAgora() + " - Pré-processamento - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos")
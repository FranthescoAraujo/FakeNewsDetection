import json
import time
import os
import csv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from DocumentRepresentationWord2Int import DocumentRepresentationWord2Int
from Classifiers import Classifiers
from datetime import datetime
from sklearn.model_selection import train_test_split

def tempoAgora():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def apagarResults(apagar):
    if (not apagar):
        csvFile = open("results/resultsTensor.csv", "r")
        leitor = csv.reader(csvFile)
        resultsCsv = []
        for linha in leitor:
            resultsCsv.append(linha)
        csvFile.close()
        lastLine = resultsCsv[-1]
        return resultsCsv, lastLine
    resultsCsv = [["LOCAL IMAGENS", "DATASET", "REMOVE STOP WORDS", "NATURAL LANGUAGE PROCESSING", "VECTOR SIZE", "CLASSIFIER", "CLASSIFIER SIZE", "MATRIX SIZE", "ACCURACY AVERAGE", "ACCURACY STANDARD DEVIATION", "PRECISION AVERAGE", "PRECISION STANDARD DEVIATION", "RECALL AVERAGE", "RECALL STANDARD DEVIATION", "AUC-PR AVERAGE", "AUC-PR STANDARD DEVIATION"]]
    lastLine = resultsCsv
    return resultsCsv, lastLine

LOCAL_PATH = "E:\\FakeNewsDetection\\NaturalLanguageProcessing\\"
apagarTudo = True

dataSetCsv = ["português"]
removeStopWordsCsv = [False]
naturalLanguageProcessingCsv = ["Tensorflow Embedding"]
vectorSizeCsv = [10]
classifierCsv = ["LSTM With Embedding"]
classifierSizeCsv = [10]
matrixSizeCsv = [10]

continueCsv = [True] * 7
if apagarTudo:
    continueCsv = [False] * 7
resultsCsv, lastLine = apagarResults(apagarTudo)
csvFile = open("results/resultsTensor.csv", "w")
results = csv.writer(csvFile)
for linha in resultsCsv:
    results.writerow(linha)
log = open("results/log.txt", "a")
del resultsCsv

for dataset in dataSetCsv:
    if (continueCsv[0] and dataset != lastLine[1]):
        continue
    continueCsv[0] = False
    for removeStopWords in removeStopWordsCsv:
        if (continueCsv[1] and removeStopWords != bool(lastLine[2])):
            continue
        continueCsv[1] = False
        tic = time.time()
        CORPUS_PATH = "../Json/inglês/" + "stopWords-" + str(removeStopWords) + "/dataset.json"
        if dataset == "português":
            CORPUS_PATH = "../Json/português/" + "stopWords-" + str(removeStopWords) + "/dataset.json"
        f = open(CORPUS_PATH, "r")
        data = json.load(f)
        f.close()
        listNews = data["dataset"]
        listLabels = data["labels"]
        listNewsTrain, listNewsTest, yTrain, yTest = train_test_split(np.array(listNews), np.array(listLabels), test_size=0.3, random_state=42)
        del listNews, listLabels
        toc = time.time() - tic
        log.write(tempoAgora() + " - Carregando dataset " + dataset + " - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos\n")
        print(tempoAgora() + " - Carregando dataset " + dataset + " - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos")

        for nlp in naturalLanguageProcessingCsv:
            if (continueCsv[2] and nlp != lastLine[3]):
                continue
            continueCsv[2] = False
            for vectorSize in vectorSizeCsv:
                if (continueCsv[3] and vectorSize != int(lastLine[4])):
                    continue
                continueCsv[3] = False
                if (nlp == "Tensorflow Embedding"):
                    for classifier in classifierCsv:
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        if classifier == "LSTM With Embedding":
                            for matrixSize in matrixSizeCsv:
                                if (continueCsv[6] and matrixSize != int(lastLine[7])):
                                    continue
                                continueCsv[6] = False
                                tic = time.time()
                                word2int = DocumentRepresentationWord2Int(listNewsTrain, listNewsTest)
                                xTrain, xTest, inputDim = word2int.intDocumentRepresentation(matrixSize=matrixSize, topWords=5000)
                                toc = time.time() - tic
                                log.write("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                print("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")
                                for classifierSize in classifierSizeCsv:
                                    if (continueCsv[5] and classifierSize != int(lastLine[6])):
                                        continue
                                    if (continueCsv[5] and classifierSize == int(lastLine[6])):
                                        continueCsv[5] = False
                                        continue  
                                    tic = time.time()
                                    classificador = Classifiers(xTrain, xTest, yTrain, yTest)
                                    classificador.setTitle(nlp + " - vector size = " + str(vectorSize) + " - matrix size = " + str(matrixSize) + " - " + classifier + " - output size = " + str(classifierSize))
                                    classificador.setLocalSave("results/" + dataset + "/stopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize))
                                    metrics = classificador.longShortTermMemoryWithEmbedding(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize, input_dim=inputDim)
                                    hiperlink = LOCAL_PATH + "results\\" + dataset + "\\stopWords-" + str(removeStopWords) + "\\" + classifier + "\\" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
                                    results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")

csvFile.close()
log.close()
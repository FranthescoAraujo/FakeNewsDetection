import time
import os
import csv
import numpy as np
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
from DocumentRepresentationWord2Int import DocumentRepresentationWord2Int
from Classifiers import Classifiers
from datetime import datetime
from sklearn.model_selection import train_test_split

def tempoAgora():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def returnLastLine(continueCsv):
    if not os.path.exists(RESULT_PATH + "results.csv"):
        csvFile = open(RESULT_PATH + "results.csv", "a")
        leitor = csv.writer(csvFile)
        leitor.writerow(["LOCAL IMAGENS", "DATASET", "REMOVE STOP WORDS", "NATURAL LANGUAGE PROCESSING", "VECTOR SIZE", "CLASSIFIER", "CLASSIFIER SIZE", "MATRIX SIZE", "ACCURACY AVERAGE", "ACCURACY STANDARD DEVIATION", "PRECISION AVERAGE", "PRECISION STANDARD DEVIATION", "RECALL AVERAGE", "RECALL STANDARD DEVIATION", "AUC-PR AVERAGE", "AUC-PR STANDARD DEVIATION"])
        csvFile.close()
        lastLine = [""] * 8
        continueCsv = [False] * 7
        return continueCsv, lastLine
    csvFile = open(RESULT_PATH + "results.csv", "r")
    leitor = csv.reader(csvFile)
    resultsCsv = []
    for linha in leitor:
        resultsCsv.append(linha)
    csvFile.close()
    if len(resultsCsv) <= 1:
        csvFile = open(RESULT_PATH + "results.csv", "w")
        leitor = csv.writer(csvFile)
        leitor.writerow(["LOCAL IMAGENS", "DATASET", "REMOVE STOP WORDS", "NATURAL LANGUAGE PROCESSING", "VECTOR SIZE", "CLASSIFIER", "CLASSIFIER SIZE", "MATRIX SIZE", "ACCURACY AVERAGE", "ACCURACY STANDARD DEVIATION", "PRECISION AVERAGE", "PRECISION STANDARD DEVIATION", "RECALL AVERAGE", "RECALL STANDARD DEVIATION", "AUC-PR AVERAGE", "AUC-PR STANDARD DEVIATION"])
        csvFile.close()
        lastLine = [""] * 8
        continueCsv = [False] * 7
        return continueCsv, lastLine
    lastLine = resultsCsv[-1]
    return continueCsv, lastLine

LOCAL_PATH = "E:/FakeNewsDetection/NaturalLanguageProcessing/"
RESULT_PATH = "Results/"
apagarTudo = False

# dataSetCsv = ["Português", "Inglês"]
# removeStopWordsCsv = [True, False]
# naturalLanguageProcessingCsv = ["Doc2vec - PV-DM", "Doc2vec - PV-DBOW", "Doc2vec - Concatenated",
#                                 "Word2vec - Skipgram - Sum", "Word2vec - Skipgram - Average", "Word2vec - CBOW - Sum", "Word2vec - CBOW - Average",
#                                 "Word2vec - Skipgram - Matrix", "Word2vec - CBOW - Matrix", "Word2vec - Skipgram - Matrix Transposed", "Word2vec - CBOW - Matrix Transposed",
#                                 "Tensorflow Embedding"]
# vectorSizeCsv = [20, 30, 40, 100, 200, 300]
# classifierCsv = ["SVM", "Naive Bayes", "RNA", "LSTM", "LSTM With Embedding"]
# classifierSizeCsv = [10, 50, 100]
# matrixSizeCsv = [10, 50, 100, 300, 400, 500]

# dataSetCsv = ["Português"]
# removeStopWordsCsv = [True, False]
# naturalLanguageProcessingCsv = ["Doc2vec - PV-DM", "Doc2vec - PV-DBOW", "Doc2vec - Concatenated",
#                         cd        "Word2vec - Skipgram - Sum", "Word2vec - Skipgram - Average", "Word2vec - CBOW - Sum", "Word2vec - CBOW - Average",
#                                 "Word2vec - Skipgram - Matrix", "Word2vec - CBOW - Matrix", "Word2vec - Skipgram - Matrix Transposed", "Word2vec - CBOW - Matrix Transposed",
#                                 "Tensorflow Embedding"]
# vectorSizeCsv = [100, 200, 300]
# classifierCsv = ["SVM", "Naive Bayes", "RNA", "LSTM", "LSTM With Embedding"]
# classifierSizeCsv = [10, 50, 100]
# matrixSizeCsv = [10, 50, 100]

dataSetCsv = ["Português"]
removeStopWordsCsv = [True, False]
naturalLanguageProcessingCsv = ["Word2vec - Skipgram - Matrix", "Word2vec - CBOW - Matrix", "Word2vec - Skipgram - Matrix Transposed", "Word2vec - CBOW - Matrix Transposed",
                                "Tensorflow Embedding"]
vectorSizeCsv = [20, 30, 40]
classifierCsv = ["LSTM", "LSTM With Embedding"]
classifierSizeCsv = [10, 50, 100]
matrixSizeCsv = [300, 400, 500]

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

continueCsv = [True] * 7
if apagarTudo:
    continueCsv = [False] * 7
    if os.path.exists(RESULT_PATH + "log.txt"):
        os.remove(RESULT_PATH + "log.txt")
    if os.path.exists(RESULT_PATH + "results.csv"):
        os.remove(RESULT_PATH + "results.csv")

continueCsv, lastLine = returnLastLine(continueCsv)
csvFile = open(RESULT_PATH + "results.csv", "a")
log = open(RESULT_PATH + "log.txt", "a")
results = csv.writer(csvFile)

for dataset in dataSetCsv:
    if (continueCsv[0] and dataset != lastLine[1]):
        continue
    continueCsv[0] = False
    for removeStopWords in removeStopWordsCsv:
        if (continueCsv[1] and str(removeStopWords) != lastLine[2]):
            continue
        continueCsv[1] = False
        tic = time.time()
        CORPUS_PATH_TRAIN = "../Json/Inglês/" + "RemoveStopWords-" + str(removeStopWords) + "/datasetTrain.json"
        CORPUS_PATH_TEST = "../Json/Inglês/" + "RemoveStopWords-" + str(removeStopWords) + "/datasetTest.json"
        if dataset == "Português":
            CORPUS_PATH_TRAIN = "../Json/Português/" + "RemoveStopWords-" + str(removeStopWords) + "/datasetTrain.json"
            CORPUS_PATH_TEST = "../Json/Português/" + "RemoveStopWords-" + str(removeStopWords) + "/datasetTest.json"
        f = open(CORPUS_PATH_TRAIN, "r")
        dataTrain = json.load(f)
        f.close()
        listNewsTrain = dataTrain["dataset"]
        yTrain = dataTrain["labels"]
        del dataTrain
        f = open(CORPUS_PATH_TEST, "r")
        dataTest = json.load(f)
        f.close()
        listNewsTest = dataTest["dataset"]
        yTest = dataTest["labels"]
        del dataTest
        toc = time.time() - tic
        log.write(tempoAgora() + " - Carregando dataset " + dataset + " - RemoveStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos\n")
        print(tempoAgora() + " - Carregando dataset " + dataset + " - RemoveStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos")

        for nlp in naturalLanguageProcessingCsv:
            if (continueCsv[2] and nlp != lastLine[3]):
                continue
            continueCsv[2] = False
            for vectorSize in vectorSizeCsv:
                if (continueCsv[3] and vectorSize != int(lastLine[4])):
                    continue
                continueCsv[3] = False
                tic = time.time()
                if (nlp == "Doc2vec - PV-DM"):
                    doc2vec = DocumentRepresentationDoc2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = doc2vec.paragraphVectorDistributedMemory(vector_size=vectorSize)
                    doc2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
                if (nlp == "Doc2vec - PV-DBOW"):
                    doc2vec = DocumentRepresentationDoc2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = doc2vec.paragraphVectorDistributedBagOfWords(vector_size=vectorSize)
                    doc2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
                if (nlp == "Doc2vec - Concatenated"):
                    doc2vec = DocumentRepresentationDoc2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = doc2vec.concatBothParagraphVectors(vector_size=vectorSize)
                    doc2vec.salvarConcat(dataset, removeStopWords, nlp, vectorSize)
                if (nlp == "Word2vec - Skipgram - Sum"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.skipGramDocumentRepresentation(meanSumOrConcat=1, vector_size=vectorSize)
                    word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
                if (nlp == "Word2vec - Skipgram - Average"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.skipGramDocumentRepresentation(meanSumOrConcat=0, vector_size=vectorSize)
                    word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
                if (nlp == "Word2vec - CBOW - Sum"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=1, vector_size=vectorSize)
                    word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
                if (nlp == "Word2vec - CBOW - Average"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=0, vector_size=vectorSize)
                    word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
                if (nlp == "Word2vec - Skipgram - Matrix"):
                    for classifier in classifierCsv:
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        if classifier == "LSTM":
                            for matrixSize in matrixSizeCsv:
                                if (continueCsv[6] and matrixSize != int(lastLine[7])):
                                    continue
                                continueCsv[6] = False
                                tic = time.time()
                                word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                                xTrain, xTest = word2vec.skipGramMatrixDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
                                word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
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
                                    classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize))
                                    metrics = classificador.longShortTermMemory(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize)
                                    classificador.salvarTensorflowLSTM(classifier, classifierSize, matrixSize, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize))
                                    hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
                                    results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")
                    
                    continue
                if (nlp == "Word2vec - CBOW - Matrix"):
                    for classifier in classifierCsv:
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        if classifier == "LSTM":
                            for matrixSize in matrixSizeCsv:
                                if (continueCsv[6] and matrixSize != int(lastLine[7])):
                                    continue
                                continueCsv[6] = False
                                tic = time.time()
                                word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                                xTrain, xTest = word2vec.continuousBagOfWordsMatrixDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
                                word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
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
                                    classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize))
                                    metrics = classificador.longShortTermMemory(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize)
                                    classificador.salvarTensorflowLSTM(classifier, classifierSize, matrixSize, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize))
                                    hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
                                    results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")      
                    
                    continue
                if (nlp == "Word2vec - Skipgram - Matrix Transposed"):
                    for classifier in classifierCsv:
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        if classifier == "LSTM":
                            for matrixSize in matrixSizeCsv:
                                if (continueCsv[6] and matrixSize != int(lastLine[7])):
                                    continue
                                continueCsv[6] = False
                                tic = time.time()
                                word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                                xTrain, xTest = word2vec.skipGramMatrixTransposedDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
                                word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
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
                                    classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize))
                                    metrics = classificador.longShortTermMemory(vector_size=matrixSize, lstm_size=classifierSize, matrix_size=vectorSize)
                                    classificador.salvarTensorflowLSTM(classifier, classifierSize, matrixSize, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize))
                                    hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
                                    results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")
                    
                    continue
                if (nlp == "Word2vec - CBOW - Matrix Transposed"):
                    for classifier in classifierCsv:
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        if classifier == "LSTM":
                            for matrixSize in matrixSizeCsv:
                                if (continueCsv[6] and matrixSize != int(lastLine[7])):
                                    continue
                                continueCsv[6] = False
                                tic = time.time()
                                word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                                xTrain, xTest = word2vec.continuousBagOfWordsMatrixTransposedDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
                                word2vec.salvar(dataset, removeStopWords, nlp, vectorSize)
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
                                    classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize))
                                    metrics = classificador.longShortTermMemory(vector_size=matrixSize, lstm_size=classifierSize, matrix_size=vectorSize)
                                    classificador.salvarTensorflowLSTM(classifier, classifierSize, matrixSize, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize))
                                    hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
                                    results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")      
                    
                    continue
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
                                word2int.salvar(dataset, removeStopWords, nlp, matrixSize)
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
                                    classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize))
                                    metrics = classificador.longShortTermMemoryWithEmbedding(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize, input_dim=inputDim)
                                    classificador.salvarTensorflowLSTMWithEmbedding(classifier, classifierSize, vectorSize, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/MatrixSize-" + str(matrixSize))
                                    hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
                                    results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")

                    continue
                toc = time.time() - tic
                log.write("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - " + str(round(toc,2)) + " segundos\n")
                print("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - " + str(round(toc,2)) + " segundos")
                for classifier in classifierCsv:
                    if classifier == "SVM":
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        if (continueCsv[4] and classifier == lastLine[5]):
                            continueCsv[4] = False
                            continueCsv[5] = False
                            continueCsv[6] = False
                            continue
                        continueCsv[5] = False
                        continueCsv[6] = False
                        tic = time.time()
                        classificador = Classifiers(xTrain, xTest, yTrain, yTest)
                        classificador.setTitle(nlp + " - vector size = " + str(vectorSize) + " - " + classifier)
                        classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize))
                        metrics = classificador.supportVectorMachine()
                        classificador.salvarSklearn(classifier, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize))
                        hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + ".png"
                        results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, "0", "0", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                        toc = time.time() - tic
                        log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos\n")
                        print("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos")

                        continue

                    if classifier == "Naive Bayes":
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        if (continueCsv[4] and classifier == lastLine[5]):
                            continueCsv[4] = False
                            continueCsv[5] = False
                            continueCsv[6] = False
                            continue
                        continueCsv[5] = False
                        continueCsv[6] = False
                        tic = time.time()
                        classificador = Classifiers(xTrain, xTest, yTrain, yTest)
                        classificador.setTitle(nlp + " - vector size = " + str(vectorSize) + " - " + classifier)
                        classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize))
                        metrics = classificador.naiveBayes()
                        classificador.salvarSklearn(classifier, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize))
                        hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + ".png"
                        results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, "0", "0", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                        toc = time.time() - tic
                        log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos\n")
                        print("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos")

                        continue

                    if classifier == "RNA":
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        for classifierSize in classifierSizeCsv:
                            if (continueCsv[5] and classifierSize != int(lastLine[6])):
                                continue
                            if (continueCsv[5] and classifierSize == int(lastLine[6])):
                                continueCsv[5] = False
                                continueCsv[6] = False
                                continue
                            continueCsv[6] = False
                            tic = time.time()
                            classificador = Classifiers(xTrain, xTest, yTrain, yTest)
                            classificador.setTitle(nlp + " - vector size = " + str(vectorSize) + " - " + classifier + " - hidden layer = " + str(classifierSize))
                            classificador.setLocalSave("Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier, "/" + nlp + " - vectorSize-" + str(vectorSize) + " - hiddenLayer-" + str(classifierSize))
                            metrics = classificador.neuralNetwork(input_size=vectorSize, hidden_layer=classifierSize)
                            classificador.salvarTensorflowRNA(classifier, classifierSize, "/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize))
                            hiperlink = LOCAL_PATH + "Results/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - hiddenLayer-" + str(classifierSize) + ".png"
                            results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, "0", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                            toc = time.time() - tic
                            log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - " + str(round(toc,2)) + " segundos\n")
                            print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - " + str(round(toc,2)) + " segundos")

csvFile.close()
log.close()
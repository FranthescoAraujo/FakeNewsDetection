import time
import os
import csv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
from Classifiers import Classifiers
from datetime import datetime
from sklearn.model_selection import train_test_split

def tempoAgora():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def apagarResults(apagar):
    if (not apagar):
        csvFile = open("results/results.csv", "r")
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

dataSetCsv = ["inglês"]
removeStopWordsCsv = [True, False]
naturalLanguageProcessingCsv = ["Doc2vec - PV-DM", "Doc2vec - PV-DBOW", "Doc2vec - Concatenated",
                                "Word2vec - Skipgram - Sum", "Word2vec - Skipgram - Average", "Word2vec - CBOW - Sum", "Word2vec - CBOW - Average",
                                "Word2vec - Skipgram - Matrix", "Word2vec - CBOW - Matrix"]
vectorSizeCsv = [100, 200, 300]
classifierCsv = ["SVM", "Naive Bayes", "RNA", "LSTM", "LSTM - Transposed"]
classifierSizeCsv = [10, 50, 100]
matrixSizeCsv = [10, 50, 100]

continueCsv = [True] * 7
if apagarTudo:
    continueCsv = [False] * 7
resultsCsv, lastLine = apagarResults(apagarTudo)
csvFile = open("results/results.csv", "w")
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
        log.write(tempoAgora() + " - Carregando dataset " + dataset + " - " + str(round(toc,2)) + " segundos\n")
        print(tempoAgora() + " - Carregando dataset " + dataset + " - " + str(round(toc,2)) + " segundos")

        tic = time.time()
        newlistNews = PreProcessing.removeAccentuation(listNews)
        newlistNews = PreProcessing.removeSpecialCharacters(newlistNews)
        newlistNews = PreProcessing.removeNumerals(newlistNews)
        newlistNews = PreProcessing.toLowerCase(newlistNews)
        if removeStopWords:
            newlistNews = PreProcessing.removeStopWords(newlistNews, dataset)
        newlistNews, listLabels = PreProcessing.removeDocumentsWithFewWords(newlistNews, listLabels)
        listNewsTrain, listNewsTest, yTrain, yTest = train_test_split(np.array(newlistNews), np.array(listLabels), test_size=0.3, random_state=42)
        del newlistNews, listLabels
        toc = time.time() - tic
        log.write(" " + tempoAgora() + " - Pré-processamento - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos\n")
        print(" " + tempoAgora() + " - Pré-processamento - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos")

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
                if (nlp == "Doc2vec - PV-DBOW"):
                    doc2vec = DocumentRepresentationDoc2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = doc2vec.paragraphVectorDistributedBagOfWords(vector_size=vectorSize)
                if (nlp == "Doc2vec - Concatenated"):
                    doc2vec = DocumentRepresentationDoc2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = doc2vec.concatBothParagraphVectors(vector_size=vectorSize)
                if (nlp == "Word2vec - Skipgram - Sum"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.skipGramDocumentRepresentation(meanSumOrConcat=1, vector_size=vectorSize)
                if (nlp == "Word2vec - Skipgram - Average"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.skipGramDocumentRepresentation(meanSumOrConcat=0, vector_size=vectorSize)
                if (nlp == "Word2vec - CBOW - Sum"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=1, vector_size=vectorSize)
                if (nlp == "Word2vec - CBOW - Average"):
                    word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                    xTrain, xTest = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=0, vector_size=vectorSize)
                if (nlp == "Word2vec - Skipgram - Matrix"):
                    for classifier in classifierCsv:
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        if classifier == "LSTM" or classifier == "LSTM - Transposed":
                            for matrixSize in matrixSizeCsv:
                                if (continueCsv[6] and matrixSize != int(lastLine[7])):
                                    continue
                                continueCsv[6] = False
                                tic = time.time()
                                word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                                xTrain, xTest = word2vec.skipGramMatrixDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
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
                                    metrics = classificador.longShortTermMemory(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize)
                                    hiperlink = LOCAL_PATH + "results\\" + dataset + "\\stopWords-" + str(removeStopWords) + "\\" + classifier + "\\" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
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
                        if classifier == "LSTM" or classifier == "LSTM - Transposed":
                            for matrixSize in matrixSizeCsv:
                                if (continueCsv[6] and matrixSize != int(lastLine[7])):
                                    continue
                                continueCsv[6] = False
                                tic = time.time()
                                word2vec = DocumentRepresentationWord2Vec(listNewsTrain, listNewsTest)
                                xTrain, xTest = word2vec.continuousBagOfWordsMatrixDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
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
                                    metrics = classificador.longShortTermMemory(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize, isTransposed=True)
                                    hiperlink = LOCAL_PATH + "results\\" + dataset + "\\stopWords-" + str(removeStopWords) + "\\" + classifier + "\\" + nlp + " - vectorSize-" + str(vectorSize) + " - outputSize-" + str(classifierSize) + " - matrixSize-" + str(matrixSize) + ".png"
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
                            continue
                        continueCsv[5] = False
                        continueCsv[6] = False
                        tic = time.time()
                        classificador = Classifiers(xTrain, xTest, yTrain, yTest)
                        classificador.setTitle(nlp + " - vector size = " + str(vectorSize) + " - " + classifier)
                        classificador.setLocalSave("results/" + dataset + "/stopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize))
                        metrics = classificador.supportVectorMachine()
                        hiperlink = LOCAL_PATH + "results\\" + dataset + "\\stopWords-" + str(removeStopWords) + "\\" + classifier + "\\" + nlp + " - vectorSize-" + str(vectorSize) + ".png"
                        results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, "-", "-", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                        toc = time.time() - tic
                        log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos\n")
                        print("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos")

                    if classifier == "Naive Bayes":
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        if (continueCsv[4] and classifier == lastLine[5]):
                            continueCsv[4] = False
                            continue
                        continueCsv[5] = False
                        continueCsv[6] = False
                        tic = time.time()
                        classificador = Classifiers(xTrain, xTest, yTrain, yTest)
                        classificador.setTitle(nlp + " - vector size = " + str(vectorSize) + " - " + classifier)
                        classificador.setLocalSave("results/" + dataset + "/stopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize))
                        metrics = classificador.naiveBayes()
                        hiperlink = LOCAL_PATH + "results\\" + dataset + "\\stopWords-" + str(removeStopWords) + "\\" + classifier + "\\" + nlp + " - vectorSize-" + str(vectorSize) + ".png"
                        results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, "-", "-", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                        toc = time.time() - tic
                        log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos\n")
                        print("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos")

                    if classifier == "RNA":
                        if (continueCsv[4] and classifier != lastLine[5]):
                            continue
                        continueCsv[4] = False
                        for classifierSize in classifierSizeCsv:
                            if (continueCsv[5] and classifierSize != int(lastLine[6])):
                                continue
                            if (continueCsv[5] and classifierSize == int(lastLine[6])):
                                continueCsv[5] = False
                                continue
                            continueCsv[6] = False
                            tic = time.time()
                            classificador = Classifiers(xTrain, xTest, yTrain, yTest)
                            classificador.setTitle(nlp + " - vector size = " + str(vectorSize) + " - " + classifier + " - hidden layer = " + str(classifierSize))
                            classificador.setLocalSave("results/" + dataset + "/stopWords-" + str(removeStopWords) + "/" + classifier + "/" + nlp + " - vectorSize-" + str(vectorSize) + " - hiddenLayer-" + str(classifierSize))
                            metrics = classificador.neuralNetwork(input_size=vectorSize, hidden_layer=classifierSize)
                            hiperlink = LOCAL_PATH + "results\\" + dataset + "\\stopWords-" + str(removeStopWords) + "\\" + classifier + "\\" + nlp + " - vectorSize-" + str(vectorSize) + " - hiddenLayer-" + str(classifierSize) + ".png"
                            results.writerow([hiperlink, dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, "-", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                            toc = time.time() - tic
                            log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - " + str(round(toc,2)) + " segundos\n")
                            print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - " + str(round(toc,2)) + " segundos")

csvFile.close()
log.close()
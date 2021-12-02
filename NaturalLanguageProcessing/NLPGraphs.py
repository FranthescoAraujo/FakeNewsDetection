import time
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
from Classifiers import Classifiers
from datetime import datetime

def tempoAgora():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

dataSetCsv = ["português", "inglês"]
removeStopWordsCsv = [True, False]
naturalLanguageProcessingCsv = ["Doc2vec - PV-DM", "Doc2vec - PV-DBOW", "Doc2vec - Concatenated",
                                "Word2vec - Skipgram - Sum", "Word2vec - Skipgram - Average", "Word2vec - CBOW - Sum", "Word2vec - CBOW - Average",
                                "Word2vec - Skipgram - Matrix", "Word2vec - CBOW - Matrix"]
vectorSizeCsv = [100, 200, 300]
classifierCsv = ["SVM", "Naive Bayes", "RNA", "LSTM", "LSTM - Transposed"]
classifierSizeCsv = [10, 50, 100]
matrixSizeCsv = [10, 50, 100]

csvFile = open("results/results.csv", "w")
results = csv.writer(csvFile)
results.writerow(["DATASET", "REMOVE STOP WORDS", "NATURAL LANGUAGE PROCESSING", "VECTOR SIZE", "CLASSIFIER", "CLASSIFIER SIZE", "MATRIX SIZE", "ACCURACY AVERAGE", "ACCURACY STANDARD DEVIATION", "PRECISION AVERAGE", "PRECISION STANDARD DEVIATION", "RECALL AVERAGE", "RECALL STANDARD DEVIATION", "AUC-PR AVERAGE", "AUC-PR STANDARD DEVIATION"])
log = open("results/log.txt", "w")

for dataset in dataSetCsv:
    for removeStopWords in removeStopWordsCsv:
        tic = time.time()
        CORPUS_PATH = "../Corpus/Ingles/"
        if dataset == "português":
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
        toc = time.time() - tic
        log.write(" " + tempoAgora() + " - Pré-processamento - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos\n")
        print(" " + tempoAgora() + " - Pré-processamento - removeStopWords = " + str(removeStopWords) + " - " + str(round(toc,2)) + " segundos")

        for nlp in naturalLanguageProcessingCsv:
            for vectorSize in vectorSizeCsv:
                tic = time.time()
                if (nlp == "Doc2vec - PV-DM"):
                    doc2vec = DocumentRepresentationDoc2Vec(newlistNews)
                    listVectors = doc2vec.paragraphVectorDistributedMemory(vector_size=vectorSize)
                if (nlp == "Doc2vec - PV-DBOW"):
                    doc2vec = DocumentRepresentationDoc2Vec(newlistNews)
                    listVectors = doc2vec.paragraphVectorDistributedBagOfWords(vector_size=vectorSize)
                if (nlp == "Doc2vec - Concatenated"):
                    doc2vec = DocumentRepresentationDoc2Vec(newlistNews)
                    listVectors = doc2vec.concatBothParagraphVectors(vector_size=vectorSize)
                if (nlp == "Word2vec - Skipgram - Sum"):
                    word2vec = DocumentRepresentationWord2Vec(newlistNews)
                    listVectors = word2vec.skipGramDocumentRepresentation(meanSumOrConcat=1, vector_size=vectorSize)
                if (nlp == "Word2vec - Skipgram - Average"):
                    word2vec = DocumentRepresentationWord2Vec(newlistNews)
                    listVectors = word2vec.skipGramDocumentRepresentation(meanSumOrConcat=0, vector_size=vectorSize)
                if (nlp == "Word2vec - CBOW - Sum"):
                    word2vec = DocumentRepresentationWord2Vec(newlistNews)
                    listVectors = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=1, vector_size=vectorSize)
                if (nlp == "Word2vec - CBOW - Average"):
                    word2vec = DocumentRepresentationWord2Vec(newlistNews)
                    listVectors = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=0, vector_size=vectorSize)
                if (nlp == "Word2vec - Skipgram - Matrix"):
                    for classifier in classifierCsv:
                        if classifier == "LSTM" or classifier == "LSTM - Transposed":
                            for matrixSize in matrixSizeCsv:
                                word2vec = DocumentRepresentationWord2Vec(newlistNews)
                                listVectors = word2vec.skipGramMatrixDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
                                toc = time.time() - tic
                                log.write("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                print("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")
                                for classifierSize in classifierSizeCsv:
                                    tic = time.time()
                                    classificador = Classifiers(listVectors, listLabels)
                                    classificador.longShortTermMemory(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize)
                                    results.writerow([dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")

                    continue
                if (nlp == "Word2vec - CBOW - Matrix"):
                    for classifier in classifierCsv:
                        if classifier == "LSTM" or classifier == "LSTM - Transposed":
                            for matrixSize in matrixSizeCsv:
                                word2vec = DocumentRepresentationWord2Vec(newlistNews)
                                listVectors = word2vec.continuousBagOfWordsMatrixDocumentRepresentation(vector_size=vectorSize, matrix_size=matrixSize)
                                toc = time.time() - tic
                                log.write("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                print("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")
                                for classifierSize in classifierSizeCsv:
                                    tic = time.time()
                                    classificador = Classifiers(listVectors, listLabels)
                                    classificador.longShortTermMemory(vector_size=vectorSize, lstm_size=classifierSize, matrix_size=matrixSize, isTransposed=True)
                                    results.writerow([dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, matrixSize, metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                                    toc = time.time() - tic
                                    log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos\n")
                                    print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - matrix size = " + str(matrixSize) + " - " + str(round(toc,2)) + " segundos")
                                    
                    continue
                toc = time.time() - tic
                log.write("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - " + str(round(toc,2)) + " segundos\n")
                print("  " + tempoAgora() + " - Processamento de Linguagem Natural - " + nlp + " - vector size = " + str(vectorSize) + " - " + str(round(toc,2)) + " segundos")

                for classifier in classifierCsv:
                    if classifier == "SVM":
                        tic = time.time()
                        classificador = Classifiers(listVectors, listLabels)
                        metrics = classificador.supportVectorMachine(vectorSize=vectorSize)
                        results.writerow([dataset, removeStopWords, nlp, vectorSize, classifier, "-", "-", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                        toc = time.time() - tic
                        log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos\n")
                        print("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos")

                    if classifier == "Naive Bayes":
                        tic = time.time()
                        classificador = Classifiers(listVectors, listLabels)
                        metrics = classificador.naiveBayes(vectorSize=vectorSize)
                        results.writerow([dataset, removeStopWords, nlp, vectorSize, classifier, "-", "-", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                        toc = time.time() - tic
                        log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos\n")
                        print("    " + tempoAgora() + " - Classificação - " + classifier + " - " + str(round(toc,2)) + " segundos")

                    if classifier == "RNA":
                        for classifierSize in classifierSizeCsv:
                            tic = time.time()
                            classificador = Classifiers(listVectors, listLabels)
                            classificador.neuralNetwork(input_size=vectorSize, hidden_layer=classifierSize)
                            results.writerow([dataset, removeStopWords, nlp, vectorSize, classifier, classifierSize, "-", metrics["accuracy"][0], metrics["accuracy"][1], metrics["precision"][0], metrics["precision"][1], metrics["recall"][0], metrics["recall"][1], metrics["AUC"][0], metrics["AUC"][1]])
                            toc = time.time() - tic
                            log.write("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - " + str(round(toc,2)) + " segundos\n")
                            print("    " + tempoAgora() + " - Classificação - " + classifier + " - classifier size = " + str(classifierSize) + " - " + str(round(toc,2)) + " segundos")

csvFile.close()
log.close()
from PreProcessing import PreProcessing
from DocumentRepresentationDoc2Vec import DocumentRepresentationDoc2Vec
from DocumentRepresentationWord2Vec import DocumentRepresentationWord2Vec
from Classifiers import Classifiers
from ClassifiersMetrics import ClassifiersMetrics
import time
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

CORPUS_PATH = "../Corpus/"
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

newlistNews = PreProcessing.removeAccentuation(listNews)
newlistNews = PreProcessing.removeSpecialCharacters(newlistNews)
newlistNews = PreProcessing.removeNumerals(newlistNews)
newlistNews = PreProcessing.toLowerCase(newlistNews)

neuralNetwork = ClassifiersMetrics()
supportVectorMachine = ClassifiersMetrics()
naiveBayes = ClassifiersMetrics()

for vectorSize in range(1,400):
    tic = time.time()
    doc2vec = DocumentRepresentationDoc2Vec(newlistNews)
    listRepresentation = doc2vec.paragraphVectorDistributedMemory(vector_size=vectorSize)
    # listRepresentation = doc2vec.paragraphVectorDistributedBagOfWords()
    # listRepresentation = doc2vec.concatBothParagraphVectors()

    # word2vec = DocumentRepresentationWord2Vec(newlistNews)
    # listRepresentation = word2vec.skipGramDocumentRepresentation()
    # listRepresentation = word2vec.continuousBagOfWordsDocumentRepresentation(meanSumOrConcat=1)

    npList = []
    for document in listRepresentation:
        npList.append(np.array(document))
    npListLabel = []
    for label in listLabel:
        npListLabel.append(np.array(label))
    npList = np.array(npList)
    npListLabel = np.array(npListLabel)

    X_train, X_test, y_train, y_test = train_test_split(npList, npListLabel, test_size=0.2, random_state=42)

    classificador = Classifiers(X_train, y_train, X_test, y_test)

    y_pred = classificador.neuralNetwork(input_size=vectorSize, hidden_layer=20)
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('results/neuralNetwork/ConfusionMatrix-Doc2vec-paragraphVectorDistributedMemory'+ str(vectorSize) +'.png')
    plt.close()
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred)
    metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    neuralNetwork.AUCList.append(metrics.auc(fpr, tpr))
    neuralNetwork.precisionList.append(neuralNetwork.precision(y_test, y_pred))
    neuralNetwork.recallList.append(neuralNetwork.recall(y_test, y_pred))
    neuralNetwork.accuracyList.append(neuralNetwork.accuracy(y_test, y_pred))
    plt.suptitle("Doc2vec - Paragraph Vector Distributed Memory VectorSize=" + str(vectorSize), fontsize=12)
    plt.title("AUC=" + format(metrics.auc(fpr, tpr), '.2f') + " Precision=" + format(neuralNetwork.precision(y_test, y_pred), '.2f') + " Recall=" + format(neuralNetwork.recall(y_test, y_pred), '.2f') + " Accuracy=" + format(neuralNetwork.accuracy(y_test, y_pred), '.2f'), fontsize=10)
    plt.savefig('results/neuralNetwork/PrecisionRecallCurve-Doc2vec-paragraphVectorDistributedMemory'+ str(vectorSize) +'.png')
    plt.close()

    y_pred = classificador.supportVectorMachine()
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('results/supportVectorMachine/ConfusionMatrix-Doc2vec-paragraphVectorDistributedMemory'+ str(vectorSize) +'.png')
    plt.close()
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred)
    metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    supportVectorMachine.AUCList.append(metrics.auc(fpr, tpr))
    supportVectorMachine.precisionList.append(supportVectorMachine.precision(y_test, y_pred))
    supportVectorMachine.recallList.append(supportVectorMachine.recall(y_test, y_pred))
    supportVectorMachine.accuracyList.append(supportVectorMachine.accuracy(y_test, y_pred))
    plt.suptitle("Doc2vec - Paragraph Vector Distributed Memory VectorSize=" + str(vectorSize), fontsize=12)
    plt.title("AUC=" + format(metrics.auc(fpr, tpr), '.2f') + " Precision=" + format(neuralNetwork.precision(y_test, y_pred), '.2f') + " Recall=" + format(neuralNetwork.recall(y_test, y_pred), '.2f') + " Accuracy=" + format(neuralNetwork.accuracy(y_test, y_pred), '.2f'), fontsize=10)
    plt.savefig('results/supportVectorMachine/PrecisionRecallCurve-Doc2vec-paragraphVectorDistributedMemory'+ str(vectorSize) +'.png')
    plt.close()

    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('results/naiveBayes/ConfusionMatrix-Doc2vec-paragraphVectorDistributedMemory'+ str(vectorSize) +'.png')
    plt.close()
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred)
    metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    naiveBayes.AUCList.append(metrics.auc(fpr, tpr))
    naiveBayes.precisionList.append(naiveBayes.precision(y_test, y_pred))
    naiveBayes.recallList.append(naiveBayes.recall(y_test, y_pred))
    naiveBayes.accuracyList.append(naiveBayes.accuracy(y_test, y_pred))
    plt.suptitle("Doc2vec - Paragraph Vector Distributed Memory VectorSize=" + str(vectorSize), fontsize=12)
    plt.title("AUC=" + format(metrics.auc(fpr, tpr), '.2f') + " Precision=" + format(neuralNetwork.precision(y_test, y_pred), '.2f') + " Recall=" + format(neuralNetwork.recall(y_test, y_pred), '.2f') + " Accuracy=" + format(neuralNetwork.accuracy(y_test, y_pred), '.2f'), fontsize=10)
    plt.savefig('results/naiveBayes/PrecisionRecallCurve-Doc2vec-paragraphVectorDistributedMemory'+ str(vectorSize) +'.png')
    plt.close()

    toc = time.time() - tic
    print("Doc2vec - Paragraph Vector Distributed Memory - Vector Size = " + str(vectorSize) + " - " + str(toc) + " segundos")

    

    
import json
import os
from PreProcessing import PreProcessing
from keras.preprocessing import sequence
from collections import Counter

class DocumentRepresentationWord2Int:
    def __init__(self, xTrain, xTest):
        self.xTrain = PreProcessing.toSplit(xTrain)
        self.xTest = PreProcessing.toSplit(xTest)

    def salvar(self, dataset, removeStopWords, nlp, matrixSize):
        path = "../Models/NaturalLanguageProcessing/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/MatrixSize-" + str(matrixSize)
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(path + "/model.json", "w")
        json.dump(self.word2id, f)
        f.close()

    def intDocumentRepresentation(self, topWords=0, matrixSize = 100):
        quantidadeWords = Counter()
        for document in self.xTrain:
            for word in document:
                quantidadeWords[word] += 1
        returnInputDim = len(quantidadeWords)
        if (topWords > 0):
            returnInputDim = topWords
        quantidadeWords = quantidadeWords.most_common(returnInputDim + 1)
        self.word2id = {}
        id = 1
        for word, freq in quantidadeWords:
            self.word2id[word] = id
            id += 1
        for document in self.xTrain:
            i = 0
            while i < len(document):
                if document[i] in self.word2id:
                    document[i] = self.word2id[document[i]]
                    i += 1
                    continue
                del document[i]
        for document in self.xTest:
            i = 0
            while i < len(document):
                if document[i] in self.word2id:
                    document[i] = self.word2id[document[i]]
                    i += 1
                    continue
                del document[i]
        returnXTrain = sequence.pad_sequences(self.xTrain, maxlen=matrixSize, padding='post')
        returnXTest = sequence.pad_sequences(self.xTest, maxlen=matrixSize, padding='post')
        return returnXTrain, returnXTest, returnInputDim
        
import numpy
import os
from PreProcessing import PreProcessing
from gensim.models.word2vec import Word2Vec

class DocumentRepresentationWord2Vec:
    def __init__(self, xTrain, xTest):
        self.xTrain = PreProcessing.toSplit(xTrain)
        self.xTest = PreProcessing.toSplit(xTest)

    def salvar(self, dataset, removeStopWords, nlp, vectorSize):
        path = "../Models/NaturalLanguageProcessing/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize)
        if not os.path.exists(path):
            os.makedirs(path)
        self.modelo.save(path + "/model.model")

    def skipGramDocumentRepresentation(self, window_size = 10, vector_size = 100, meanSumOrConcat = 0):
        self.modelo = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=1, min_count=1)
        model = self.modelo.wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [0]*vector_size
            numWords = len(document)
            for word in document:
                if word not in model:
                    continue
                if meanSumOrConcat == 0:
                    documentRepresentation+=model[word]
                if meanSumOrConcat == 1:
                    documentRepresentation+=model[word]
                    numWords = 1
                if meanSumOrConcat == 2:
                    numWords = 1
                    pass
            returnXTrain.append(numpy.divide(documentRepresentation,numWords).tolist())
        for document in self.xTest:
            documentRepresentation = [0]*vector_size
            numWords = len(document)
            for word in document:
                if word not in model:
                    continue
                if meanSumOrConcat == 0:
                    documentRepresentation+=model[word]
                if meanSumOrConcat == 1:
                    documentRepresentation+=model[word]
                    numWords = 1
                if meanSumOrConcat == 2:
                    numWords = 1
                    pass
            returnXTest.append(numpy.divide(documentRepresentation,numWords).tolist())
        return returnXTrain, returnXTest

    def continuousBagOfWordsDocumentRepresentation(self, window_size = 10, vector_size = 100, meanSumOrConcat = 0):
        self.modelo = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=0, min_count=1)
        model = self.modelo.wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [0]*vector_size
            numWords = len(document)
            for word in document:
                if word not in model:
                    continue
                if meanSumOrConcat == 0:
                    documentRepresentation+=model[word]
                if meanSumOrConcat == 1:
                    documentRepresentation+=model[word]
                    numWords = 1
                if meanSumOrConcat == 2:
                    numWords = 1
                    pass
            returnXTrain.append(numpy.divide(documentRepresentation,numWords).tolist())
        for document in self.xTest:
            documentRepresentation = [0]*vector_size
            numWords = len(document)
            for word in document:
                if word not in model:
                    continue
                if meanSumOrConcat == 0:
                    documentRepresentation+=model[word]
                if meanSumOrConcat == 1:
                    documentRepresentation+=model[word]
                    numWords = 1
                if meanSumOrConcat == 2:
                    numWords = 1
                    pass
            returnXTest.append(numpy.divide(documentRepresentation,numWords).tolist())    
        return returnXTrain, returnXTest
    
    def skipGramMatrixDocumentRepresentation(self, window_size = 10, vector_size = 100, matrix_size = 100):
        self.modelo = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=1, min_count=1)
        model = self.modelo.wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [[0]*vector_size]*matrix_size
            index = 0
            for word in document:
                if index >= matrix_size:
                    break
                if word in model:
                    documentRepresentation[index] = model[word]
                    index += 1
            returnXTrain.append(documentRepresentation)
        for document in self.xTest:
            documentRepresentation = [[0]*vector_size]*matrix_size
            index = 0
            for word in document:
                if index >= matrix_size:
                    break
                if word in model:
                    documentRepresentation[index] = model[word]
                    index += 1
            returnXTest.append(documentRepresentation)
        return returnXTrain, returnXTest
    
    def continuousBagOfWordsMatrixDocumentRepresentation(self, window_size = 10, vector_size = 100, matrix_size = 100):
        self.modelo = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=0, min_count=1)
        model = self.modelo.wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [[0]*vector_size]*matrix_size
            index = 0
            for word in document:
                if index >= matrix_size:
                    break
                if word in model:
                    documentRepresentation[index] = model[word]
                    index += 1
            returnXTrain.append(documentRepresentation)
        for document in self.xTest:
            documentRepresentation = [[0]*vector_size]*matrix_size
            index = 0
            for word in document:
                if index >= matrix_size:
                    break
                if word in model:
                    documentRepresentation[index] = model[word]
                    index += 1
            returnXTest.append(documentRepresentation)
        return returnXTrain, returnXTest
        
    def skipGramMatrixTransposedDocumentRepresentation(self, window_size = 10, vector_size = 100, matrix_size = 100):
        returnXTrain, returnXTest = self.skipGramMatrixDocumentRepresentation(window_size, vector_size, matrix_size)
        returnXTrain = numpy.transpose(returnXTrain, (0,2,1))
        returnXTest = numpy.transpose(returnXTest, (0,2,1))
        return returnXTrain, returnXTest

    def continuousBagOfWordsMatrixTransposedDocumentRepresentation(self, window_size = 10, vector_size = 100, matrix_size = 100):
        returnXTrain, returnXTest = self.continuousBagOfWordsMatrixDocumentRepresentation(window_size, vector_size, matrix_size)
        returnXTrain = numpy.transpose(returnXTrain, (0,2,1))
        returnXTest = numpy.transpose(returnXTest, (0,2,1))
        return returnXTrain, returnXTest
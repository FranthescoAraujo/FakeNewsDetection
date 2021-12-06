from PreProcessing import PreProcessing
from gensim.models.word2vec import Word2Vec
import numpy

class DocumentRepresentationWord2Vec:
    def __init__(self, xTrain, xTest):
        self.xTrain = PreProcessing.toSplit(xTrain)
        self.xTest = PreProcessing.toSplit(xTest)

    def skipGramDocumentRepresentation(self, window_size = 10, vector_size = 100, meanSumOrConcat = 0):
        model = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=1, min_count=1).wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [0]*vector_size
            numWords = len(document)
            for word in document:
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
        model = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=0, min_count=1).wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [0]*vector_size
            numWords = len(document)
            for word in document:
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
        model = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=1, min_count=1).wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [[0]*vector_size]*matrix_size
            for index, word in enumerate(document):
                if index >= matrix_size or index == (len(document)-1):
                    returnXTrain.append(documentRepresentation)
                    documentRepresentation = []
                    break
                if word not in model:
                    continue
                documentRepresentation[index] = model[word]
        for document in self.xTest:
            documentRepresentation = [[0]*vector_size]*matrix_size
            for index, word in enumerate(document):
                if index >= matrix_size or index == (len(document)-1):
                    returnXTest.append(documentRepresentation)
                    documentRepresentation = []
                    break
                if word not in model:
                    continue
                documentRepresentation[index] = model[word]
        return returnXTrain, returnXTest
    
    def continuousBagOfWordsMatrixDocumentRepresentation(self, window_size = 10, vector_size = 100, matrix_size = 100):
        model = Word2Vec(sentences=self.xTrain, vector_size=vector_size, window=window_size, workers=4, sg=0, min_count=1).wv
        returnXTrain = []
        returnXTest = []
        for document in self.xTrain:
            documentRepresentation = [[0]*vector_size]*matrix_size
            for index, word in enumerate(document):
                if index >= matrix_size or index == (len(document)-1):
                    returnXTrain.append(documentRepresentation)
                    documentRepresentation = []
                    break
                if word not in model:
                    continue
                documentRepresentation[index] = model[word]
        for document in self.xTest:
            documentRepresentation = [[0]*vector_size]*matrix_size
            for index, word in enumerate(document):
                if index >= matrix_size or index == (len(document)-1):
                    returnXTest.append(documentRepresentation)
                    documentRepresentation = []
                    break
                if word not in model:
                    continue
                documentRepresentation[index] = model[word]
        return returnXTrain, returnXTest
        
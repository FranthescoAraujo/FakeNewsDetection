from PreProcessing import PreProcessing
from gensim.models.word2vec import Word2Vec
import numpy

class DocumentRepresentationWord2Vec:
    documents = []

    def __init__(self, documents):
        self.documents = PreProcessing.toSplit(documents)

    def skipGramDocumentRepresentation(self, window_size = 10, vector_size = 100, meanSumOrConcat = 0):
        model = Word2Vec(sentences=self.documents, vector_size=vector_size, window=window_size, workers=4, sg=1, min_count=1).wv
        returnValue = []
        for document in self.documents:
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
            returnValue.append(numpy.divide(documentRepresentation,numWords).tolist())
        return returnValue

    def continuousBagOfWordsDocumentRepresentation(self, window_size = 10, vector_size = 100, meanSumOrConcat = 0):
        model = Word2Vec(sentences=self.documents, vector_size=vector_size, window=window_size, workers=4, sg=0, min_count=1).wv
        returnValue = []
        for document in self.documents:
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
            returnValue.append(numpy.divide(documentRepresentation,numWords).tolist())
        return returnValue
    
    def skipGramMatrixDocumentRepresentation(self, window_size = 10, vector_size = 100, meanSumOrConcat = 0, matrix_size = 100):
        model = Word2Vec(sentences=self.documents, vector_size=vector_size, window=window_size, workers=4, sg=1, min_count=1).wv
        returnValue = []
        for document in self.documents:
            documentRepresentation = [[0]*vector_size]*matrix_size
            for index, word in enumerate(document):
                if index >= matrix_size or index == (len(document)-1):
                    returnValue.append(documentRepresentation)
                    documentRepresentation = []
                    break
                documentRepresentation[index] = model[word]
        return returnValue
    
    def continuousBagOfWordsMatrixDocumentRepresentation(self, window_size = 10, vector_size = 100, meanSumOrConcat = 0, matrix_size = 100):
        model = Word2Vec(sentences=self.documents, vector_size=vector_size, window=window_size, workers=4, sg=0, min_count=1).wv
        returnValue = []
        for document in self.documents:
            documentRepresentation = [[0]*vector_size]*matrix_size
            for index, word in enumerate(document):
                if index >= matrix_size or index == (len(document)-1):
                    returnValue.append(documentRepresentation)
                    documentRepresentation = []
                    break
                documentRepresentation[index] = model[word]
        return returnValue
        
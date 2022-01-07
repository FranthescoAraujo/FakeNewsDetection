import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class DocumentRepresentationDoc2Vec:
    def __init__(self, xTrain, xTest):
        self.xTrain = [TaggedDocument(doc, [i]) for i, doc in enumerate(xTrain)]
        self.xTest = xTest

    def salvar(self, dataset, removeStopWords, nlp, vectorSize):
        path = "../Models/NaturalLanguageProcessing/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize)
        if not os.path.exists(path):
            os.makedirs(path)
        self.modelo.save(path + "/model.model")

    def salvarConcat(self, dataset, removeStopWords, nlp, vectorSize):
        path = "../Models/NaturalLanguageProcessing/" + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + nlp + "/VectorSize-" + str(vectorSize)
        if not os.path.exists(path):
            os.makedirs(path)
        self.modelo1.save(path + "/model1.model")
        self.modelo2.save(path + "/model2.model")

    def paragraphVectorDistributedMemory(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnXTrain = []
        returnXTest = []
        self.modelo = Doc2Vec(self.xTrain, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=1)
        representation = self.modelo
        for document in range(len(representation.dv)):
            returnXTrain.append(representation.dv[document].tolist())
        for document in self.xTest:
            returnXTest.append(representation.infer_vector(document.split()).tolist())
        return returnXTrain, returnXTest

    def paragraphVectorDistributedBagOfWords(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnXTrain = []
        returnXTest = []
        self.modelo = Doc2Vec(self.xTrain, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0)
        representation = self.modelo
        for document in range(len(representation.dv)):
            returnXTrain.append(representation.dv[document].tolist())
        for document in self.xTest:
            returnXTest.append(representation.infer_vector(document.split()).tolist())
        return returnXTrain, returnXTest

    def concatBothParagraphVectors(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnXTrain = []
        returnXTest = []
        self.modelo1 = Doc2Vec(self.xTrain, vector_size=(int)(vector_size/2), window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0)
        self.modelo2 = Doc2Vec(self.xTrain, vector_size=(int)(vector_size/2), window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=1)
        representation01 = self.modelo1
        representation02 = self.modelo2
        for document in range(len(representation01.dv)):
            returnXTrain.append(representation01.dv[document].tolist() + representation02.dv[document].tolist())
        for document in self.xTest:
            returnXTest.append(representation01.infer_vector(document.split()).tolist() + representation02.infer_vector(document.split()).tolist())
        return returnXTrain, returnXTest
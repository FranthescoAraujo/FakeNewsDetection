from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class DocumentRepresentationDoc2Vec:
    def __init__(self, documents):
        self.documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]

    def paragraphVectorDistributedMemory(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnValue = []
        representation = Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=1)
        for document in range(len(representation.dv)):
            returnValue.append(representation.dv[document].tolist())
        return returnValue

    def paragraphVectorDistributedBagOfWords(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnValue = []
        representation = Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0)
        for document in range(len(representation.dv)):
            returnValue.append(representation.dv[document].tolist())
        return returnValue

    def concatBothParagraphVectors(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnValue = []
        representation01 = Doc2Vec(self.documents, vector_size=(int)(vector_size/2), window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0)
        representation02 = Doc2Vec(self.documents, vector_size=(int)(vector_size/2), window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=1)
        for document in range(len(representation01.dv)):
            returnValue.append(representation01.dv[document].tolist() + representation02.dv[document].tolist())
        return returnValue
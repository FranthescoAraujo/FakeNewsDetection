from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class DocumentRepresentationDoc2Vec:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    documents = []
    def __init__(self, documents):
        self.documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]

    def paragraphVectorDistributedMemory(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnValue = []
        for document in range(len(Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=1).dv)):
            returnValue.append(Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=1).dv[document].tolist())
        return returnValue

    def paragraphVectorDistributedBagOfWords(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnValue = []
        for document in range(len(Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0).dv)):
            returnValue.append(Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0).dv[document].tolist())
        return returnValue

    def concatBothParagraphVectors(self, vector_size = 100, window_size = 10, dm_mean = 1, dm_concat = 0):
        returnValue = []
        for document in range(len(Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0).dv)):
            returnValue.append(Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=0).dv[document].tolist() + Doc2Vec(self.documents, vector_size=vector_size, window=window_size, workers=4, dm_mean=dm_mean, dm_concat=dm_concat, dm=1).dv[document].tolist())
        return returnValue
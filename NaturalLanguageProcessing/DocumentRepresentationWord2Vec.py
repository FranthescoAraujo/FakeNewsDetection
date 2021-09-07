class DocumentRepresentationWord2Vec:
    skipGramWordsRepresentation = []
    continuousBagOfWordsWordsRepresentation = []
    documents = []

    def __init__(self, documents):
        self.documents = documents

    def skipGram(window_size = 10, representation_size = 100):
        pass
    
    def continuousBagOfWords(window_size = 10, representation_size = 100):
        pass

    def skipGramDocumentRepresentation(meanSumOrConcat = 0):
        if meanSumOrConcat == 0:
            return
        if meanSumOrConcat == 1:
            return
        if meanSumOrConcat == 2:
            return

    def continuousBagOfWordsDocumentRepresentation(meanSumOrConcat = 0):
        if meanSumOrConcat == 0:
            return
        if meanSumOrConcat == 1:
            return
        if meanSumOrConcat == 2:
            return
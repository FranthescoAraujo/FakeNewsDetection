from PreProcessing import PreProcessing
from gensim.models import TfidfModel
from gensim import corpora

class TermFrequencyInverseDocumentFrequency:
    def createVectors(self, documents):
        textSplit = PreProcessing.toSplit(documents)
        dictionary = corpora.Dictionary(textSplit)
        corpus = [dictionary.doc2bow(line) for line in textSplit]
        model = TfidfModel(corpus)
        returnValue = []
        for i in range(len(model[corpus])):
            tfIdfRepresentation = [0]*len(dictionary)
            for representation in model[corpus[i]]:
                tfIdfRepresentation[representation[0]] = representation[1]
            returnValue.append(tfIdfRepresentation)    
        return returnValue
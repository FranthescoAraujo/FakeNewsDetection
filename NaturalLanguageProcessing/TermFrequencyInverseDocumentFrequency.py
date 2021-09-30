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

    def wordInDocument(self, documents):
        list = []
        with open('words.txt', 'w') as f:
            for index, document in enumerate(documents):
                for word in document.split():
                    if word not in list:
                        list.append(word)
                        f.write(str(index) + ".txt - " + word + "\n")

    def countWords(self, documents):
        dictionary = {}
        for document in documents:
            for word in document.split():
                if word in dictionary:
                    dictionary[word] += 1
                    continue
                dictionary[word] = 1
        dictionary = dict(sorted(dictionary.items(), key=lambda kv: kv[1], reverse=True))
        with open('wordsCount.txt', 'w') as f:
            for element in dictionary:
                f.write(element + " - " + str(dictionary[element]) + "\n")

                    
                
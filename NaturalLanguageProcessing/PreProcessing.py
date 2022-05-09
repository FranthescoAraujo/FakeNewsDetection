import json
import random
import nltk
import re
# nltk.download('rslp')

class PreProcessing:
    def toLowerCase(documents):
        returnValue = []
        for document in documents:
            returnValue.append(document.lower())
        return returnValue

    def removeAccentuation(documents):
        from unicodedata import normalize
        returnValue = []
        for document in documents:
            returnValue.append(normalize("NFD", document).encode("ascii", "ignore").decode("utf-8"))
        return returnValue

    def removeSpecialCharacters(documents):
        returnValue = []
        for document in documents:
            returnValue.append(re.sub('\W+', ' ', document))
        return returnValue
        
    def removeNumerals(documents):
        returnValue = []
        for document in documents:
            text = ""
            for word in document:
                if not word.isdigit():
                    text += word
            returnValue.append(text)
        return returnValue

    def removeStopWords(documents, dataset = "Português"):
        if dataset == "Português":
            return PreProcessing.__remove("StopWordsPortugues.json", documents)
        return PreProcessing.__remove("StopWordsIngles.json", documents)
    
    def removeDocumentsWithFewWords(documents, labels, numWords = 10):
        returnValueDocuments = []
        returnValueLabels = []
        for index, document in enumerate(documents):
            if len(document.split()) <= numWords:
                continue
            returnValueDocuments.append(documents[index])
            returnValueLabels.append(labels[index])
        return returnValueDocuments, returnValueLabels

    def removeDocumentsWithManyWords(documents, dataset, numWordsMin = 50, numWordsMax = 150):
        if dataset == "Português":
            returnValueDocuments = []
            for documentIndex, document in enumerate(documents):
                numWords = random.randint(numWordsMin, numWordsMax)
                documentSplit = document.split()
                quantidadePalavras = len(documentSplit)
                if (quantidadePalavras >= numWords):
                    valorInicial = random.randint(0, (quantidadePalavras - numWords))
                    documentSplit = documentSplit[valorInicial:(valorInicial + numWords)]
                text = ""
                for word in documentSplit:
                    text += word + " "
                returnValueDocuments.append(text)
            return returnValueDocuments
        return documents

    def convertWordsToStemming(documents, dataset = "Português"):
        if dataset == "Português":
            return PreProcessing.__convert(documents)
        return documents

    def __convert(documents):
        stemmer = nltk.stem.RSLPStemmer()
        returnValue = []
        for document in documents:
            text = ""
            for word in document.split():
                text += str(stemmer.stem(word)) + " "
            returnValue.append(text)
        return returnValue

    def __remove(jsonFile, documents):
        returnValue = []
        f = open(jsonFile)
        data = json.load(f)
        words = data['words']
        for document in documents:
            text = ""
            for word in document.split():
                if word not in words:
                    text += word + " "
            returnValue.append(text)
        return returnValue

    def toSplit(documents):
        returnValue = []
        for document in documents:
            returnValue.append(document.split())
        return returnValue
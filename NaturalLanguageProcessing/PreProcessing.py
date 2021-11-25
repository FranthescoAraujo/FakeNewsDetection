import json

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
        import re
        returnValue = []
        for document in documents:
            returnValue.append(re.sub('[^A-Za-z0-9\']+', ' ', document))
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

    def removeStopWords(documents, portugues = True):
        returnValue = []
        if portugues:
            f = open('StopWordsPortugues.json')
            data = json.load(f)
            words = data['words']
            for document in documents:
                text = ""
                for word in document.split():
                    if word not in words:
                        text += word + " "
                returnValue.append(text)
            return returnValue

CORPUS_PATH = "../Corpus/"

f = open(CORPUS_PATH + "train.csv", "r")
corpus = f.read()
documentNumber = 0
corpusList = []
text = ""
title = ""
aux = False
for letter in corpus:
    if letter == "\n" or aux:
        aux = True
        title += letter
        if (letter.isdigit()) or (letter == "\n"):
            continue
        if title[1:-1] == str(documentNumber):
            corpusList.append(text.strip())
            text = ""
            title = ""
            documentNumber += 1
        else:
            aux = False
            text += " "
            title = ""
    text += letter   

corpusListTrue = []
corpusListFake = []
corpusListErros = []

for text in corpusList:
    if (text[-1] == "0"):
        corpusListTrue.append(text)
    elif (text[-1] == "1"):
        corpusListFake.append(text)
    else:
        corpusListErros.append(text)

print("Quantidade de erros = " + str(len(corpusListErros)))
print("Quantidade de TrueNews = " + str(len(corpusListTrue)))
print("Quantidade de FakeNews = " + str(len(corpusListFake)))

documentNumber = 1
for text in corpusListTrue:
    with open(CORPUS_PATH + "Ingles/True/" + str(documentNumber) + ".txt", "w") as f:
        f.write(text)
    documentNumber+=1

documentNumber = 1
for text in corpusListFake:
    with open(CORPUS_PATH + "Ingles/Fake/" + str(documentNumber) + ".txt", "w") as f:
        f.write(text)
    documentNumber+=1

documentNumber = 1
for text in corpusListErros:
    with open(CORPUS_PATH + "Ingles/Erros/" + str(documentNumber) + ".txt", "w") as f:
        f.write(text)
    documentNumber+=1





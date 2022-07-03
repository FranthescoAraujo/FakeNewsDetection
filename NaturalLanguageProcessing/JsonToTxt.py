import json
import os

def salvarTxt(PATH_CORPUS_TEST, dataset, removeStopWords, fakeOrTrue, text, name):
    CORPUS_PATH = PATH_CORPUS_TEST + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/" + fakeOrTrue + "/"
    if not os.path.exists(CORPUS_PATH):
        os.makedirs(CORPUS_PATH)
    f = open(CORPUS_PATH + str(name) + ".txt", "w")
    f.write(text)
    f.close()

def openAndConvertJson(PATH_JSON_TEST, PATH_CORPUS_TEST, dataset, removeStopWords):
    CORPUS_PATH = PATH_JSON_TEST + dataset + "/RemoveStopWords-" + str(removeStopWords) + "/datasetTest.json"
    f = open(CORPUS_PATH, "r")
    dataTrain = json.load(f)
    f.close()
    listNews = dataTrain["dataset"]
    listLabels = dataTrain["labels"]
    del dataTrain
    nameFake = 1
    nameTrue = 1
    for index, text in enumerate(listNews):
        if (listLabels[index]) == 1:
            salvarTxt(PATH_CORPUS_TEST, dataset, removeStopWords, "Fake", text, nameFake)
            nameFake += 1
            continue
        salvarTxt(PATH_CORPUS_TEST, dataset, removeStopWords, "True", text, nameTrue)
        nameTrue += 1

PATH_JSON_TEST = "../JsonTest/"
PATH_CORPUS_TEST = "../CorpusTest/"
# dataSetCsv = ["Português", "Inglês"]
# removeStopWordsCsv = [True, False]
dataSetCsv = ["Português"]
removeStopWordsCsv = [True, False]

for dataset in dataSetCsv:
    for removeStopWords in removeStopWordsCsv:
        openAndConvertJson(PATH_JSON_TEST, PATH_CORPUS_TEST, dataset, removeStopWords)
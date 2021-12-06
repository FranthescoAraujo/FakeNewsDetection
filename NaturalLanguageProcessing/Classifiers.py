
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc, precision_score, recall_score, accuracy_score
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)

class Classifiers:
    def __init__(self, xTrain, xTest, yTrain, yTest, foldsNumber = 5):
        self.foldsNumber = foldsNumber
        self.xTrain, self.xTest, self.yTrain, self.yTest = np.array(xTrain), np.array(xTest), np.array(yTrain), np.array(yTest)
        self.metrics = {"accuracy":[0, 0], "precision":[0, 0], "recall":[0, 0], "AUC":[0, 0]}

    def setTitle(self, title):
        self.title = title
    
    def setLocalSave(self, localSave):
        self.localSave = localSave

    def supportVectorMachine(self):
        svm = SVC(C=1, probability=True ,random_state=42)
        return self.__crossValidationSklearn(svm)

    def naiveBayes(self):
        nb = GaussianNB()
        return self.__crossValidationSklearn(nb)

    def neuralNetwork(self, input_size = 100, hidden_layer = 100):
        callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(input_size,1)),
            keras.layers.Dense(hidden_layer, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        return self.__crossValidationTensorflow(model, callback)

    def longShortTermMemory(self, lstm_size = 100,  vector_size = 100, matrix_size = 100, isTransposed = False):
        callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
        if (isTransposed):
            model = keras.Sequential([
            keras.layers.LSTM(lstm_size, input_shape=(vector_size, matrix_size)),
            keras.layers.Dense(1, activation='sigmoid')
            ])
            self.xTrain = np.transpose(self.xTrain, (0,2,1))
            self.xTest = np.transpose(self.xTest, (0,2,1))
            return self.__crossValidationTensorflow(model, callback)
        else:
            model = keras.Sequential([
                keras.layers.LSTM(lstm_size, input_shape=(matrix_size, vector_size)),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            return self.__crossValidationTensorflow(model, callback)
        
    def __crossValidationSklearn(self, classifier):
        f, axes = plt.subplots(2, 3, figsize=(10, 5))
        plt.suptitle(self.title, fontsize=10)
        k_fold = KFold(n_splits=self.foldsNumber, shuffle=True, random_state=42)
        y_real = []
        y_proba = []
        precisionList = []
        recallList = []
        accuracyList = []
        AUCList = []
        x, y = 0, 1
        for i, (train_index, validation_index) in enumerate(k_fold.split(self.xTrain)):
            xTrain, xValidation = self.xTrain[train_index], self.xTrain[validation_index]
            yTrain, yValidation = self.yTrain[train_index], self.yTrain[validation_index]
            classifier.fit(xTrain, yTrain)
            pred_proba = classifier.predict_proba(self.xTest)
            precision, recall, _ = precision_recall_curve(self.yTest, pred_proba[:,1])
            yPred = classifier.predict(self.xTest)
            yPred = np.round(yPred).astype(int)
            confusionMatrix = confusion_matrix(self.yTest, yPred)
            ConfusionMatrixDisplay(confusionMatrix).plot(cmap=plt.cm.Blues, ax=axes[x, y])
            y += 1
            if(y == 3):
                x = 1
                y = 0
            lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            axes[0, 0].step(recall, precision, label=lab)
            y_real.append(self.yTest)
            y_proba.append(pred_proba[:,1])
            AUCList.append(auc(recall, precision))
            accuracyList.append(accuracy_score(self.yTest, yPred))
            precisionList.append(precision_score(self.yTest, yPred))
            recallList.append(recall_score(self.yTest, yPred))
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        lab = 'Overall AUC=%.4f' % (auc(recall, precision))
        axes[0, 0].step(recall, precision, label=lab, lw=2, color='black')
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].legend(loc='lower left', fontsize='small')
        f.tight_layout()
        f.savefig(self.localSave + ".png")
        plt.close(f)
        self.metrics["accuracy"][0] = str(round(np.mean(accuracyList),4))
        self.metrics["accuracy"][1] = str(round(np.std(accuracyList),4))
        self.metrics["precision"][0] = str(round(np.mean(precisionList),4))
        self.metrics["precision"][1] = str(round(np.std(precisionList),4))
        self.metrics["recall"][0] = str(round(np.mean(recallList),4))
        self.metrics["recall"][1] = str(round(np.std(recallList),4))
        self.metrics["AUC"][0] = str(round(np.mean(AUCList),4))
        self.metrics["AUC"][1] = str(round(np.std(AUCList),4))
        return self.metrics
    
    def __crossValidationTensorflow(self, classifier, callback):
        f, axes = plt.subplots(2, 3, figsize=(10, 5))
        plt.suptitle(self.title, fontsize=10)
        k_fold = KFold(n_splits=self.foldsNumber, shuffle=True, random_state=42)
        y_real = []
        y_proba = []
        precisionList = []
        recallList = []
        accuracyList = []
        AUCList = []
        x, y = 0, 1
        for i, (train_index, validation_index) in enumerate(k_fold.split(self.xTrain)):
            xTrain, xValidation = self.xTrain[train_index], self.xTrain[validation_index]
            yTrain, yValidation = self.yTrain[train_index], self.yTrain[validation_index]
            classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            classifier.fit(xTrain, yTrain, epochs=10, validation_data=(xValidation, yValidation), callbacks=[callback], verbose=0)
            yPred = classifier.predict(self.xTest)
            pred_proba = yPred
            yPred = np.round(yPred).astype(int)
            precision, recall, _ = precision_recall_curve(self.yTest, pred_proba)
            confusionMatrix = confusion_matrix(self.yTest, yPred)
            ConfusionMatrixDisplay(confusionMatrix).plot(cmap=plt.cm.Blues, ax=axes[x, y])
            y += 1
            if(y == 3):
                x = 1
                y = 0
            lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            axes[0, 0].step(recall, precision, label=lab)
            y_real.append(self.yTest)
            y_proba.append(pred_proba)
            AUCList.append(auc(recall, precision))
            accuracyList.append(accuracy_score(self.yTest, yPred))
            precisionList.append(precision_score(self.yTest, yPred))
            recallList.append(recall_score(self.yTest, yPred))
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        lab = 'Overall AUC=%.4f' % (auc(recall, precision))
        axes[0, 0].step(recall, precision, label=lab, lw=2, color='black')
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].legend(loc='lower left', fontsize='small')
        f.tight_layout()
        f.savefig(self.localSave + ".png")
        plt.close(f)
        self.metrics["accuracy"][0] = str(round(np.mean(accuracyList),4))
        self.metrics["accuracy"][1] = str(round(np.std(accuracyList),4))
        self.metrics["precision"][0] = str(round(np.mean(precisionList),4))
        self.metrics["precision"][1] = str(round(np.std(precisionList),4))
        self.metrics["recall"][0] = str(round(np.mean(recallList),4))
        self.metrics["recall"][1] = str(round(np.std(recallList),4))
        self.metrics["AUC"][0] = str(round(np.mean(AUCList),4))
        self.metrics["AUC"][1] = str(round(np.std(AUCList),4))
        return self.metrics
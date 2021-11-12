import tensorflow.keras as keras
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

class Classifiers:
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def neuralNetwork(self, input_size = 300, hidden_layer = 100):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(input_size,1)),
            keras.layers.Dense(hidden_layer, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, epochs=10, validation_data=(self.x_test, self.y_test), verbose=0)
        y_pred = model.predict(self.x_test)
        y_pred = np.round(y_pred).astype(int)
        return y_pred

    def naiveBayes(self):
        nb = GaussianNB()
        nb.fit(self.x_train, self.y_train)
        y_pred = nb.predict(self.x_test)
        return y_pred

    def supportVectorMachine(self):
        svm = SVC()
        svm.fit(self.x_train, self.y_train)
        y_pred = svm.predict(self.x_test)
        return y_pred

    def longShortTermMemory(self):
        model = keras.Sequential([
            keras.layers.Embedding(),
            keras.layers.LSTM(100),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, epochs=10, validation_data=(self.x_test, self.y_test))
        y_pred = model.predict(self.x_test)
        y_pred = np.round(y_pred).astype(int)
        return y_pred


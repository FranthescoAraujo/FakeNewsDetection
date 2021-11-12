class ClassifiersMetrics:
    accuracyList = []
    precisionList = []
    recallList = []
    AUCList = []

    def accuracy(self, y_test, y_pred):
        fp, fn, tp, tn = self.computeMetrics(y_test, y_pred)
        return (tp + tn)/(tp + tn + fp + fn)

    def precision(self, y_test, y_pred):
        fp, fn, tp, tn = self.computeMetrics(y_test, y_pred)
        return tp/(tp + fp)

    def recall(self, y_test, y_pred):
        fp, fn, tp, tn = self.computeMetrics(y_test, y_pred)
        return tp/(tp + fn)

    def computeMetrics(self, y_test, y_pred):
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for index, value in enumerate(y_test):
            if (value == 1):
                if(value == y_pred[index]):
                    tp += 1
                    continue
                fn += 1
                continue
            if (value == y_pred[index]):
                tn += 1
                continue
            fp += 1
        return fp, fn, tp, tn
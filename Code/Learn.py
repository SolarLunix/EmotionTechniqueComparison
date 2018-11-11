from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

class machineLearning():

    def __init__(self, model):
        self.model = model


    def train(self, x, y):
        self.model.fit(x,y)
        return self.model

    def test(self, x, y):
        pred = self.model.predict(x)

    def crossVal(self,x, y, splits):
        x = np.array(x)
        y = np.array(y)

        print(x.shape)
        scale = StandardScaler()
        predicted_y = []
        expected_y = []

        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=11)

        for train_index, test_index in skf.split(x, y):
            # specific ".loc" syntax for working with dataframes
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_train = scale.fit_transform(x_train,y_train)
            x_test = scale.transform(x_test)

            self.model.fit(x_train, y_train)

            # store result from classification
            predicted_y.extend(self.model.predict(x_test))

            # store expected result for this specific fold
            expected_y.extend(y_test)

        return accuracy_score(expected_y, predicted_y)
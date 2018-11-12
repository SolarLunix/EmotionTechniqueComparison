import pickle
import pandas
import datetime
import os

class SaveFeatures():

    def save(self, name, features, labels):
        savedFiles = (features, labels)
        file = open('Saved/{0}.pkl'.format(name),'wb')
        pickle._dump(savedFiles, file)

    def load(self, name):
        pickle_in = open('Saved/{0}.pkl'.format(name), "rb")
        files = pickle.load(pickle_in)
        return files[0], files[1]


class SaveRunInfo:
    def __init__(self):
        self.df = pandas.read_excel('Saved/AllTrials.xlsx', header=0)

    def savetrial(self, names, acc, times):
        date = datetime.datetime.now()

        data = {
            "Date": date,
            "Database": names["db"],
            'Extraction': names['extraction'],
            "Selection": names['selection'],
            "Classification": names['classification'],
            "Accuracy": acc,
            "Total Time": times["Total"],
            "Read and Crop Time Total": times['Read In'],
            "Extraction Time Total": times['Extract'],
            "Selection and Learning Time Total": times["Learn"],
            "Extraction Time per Fold": times["Extract fold"],
            "Learning Time per Fold": times["Learning fold"],
            "Prediction Time per Fold": times["Prediction fold"]
        }

        self.df = self.df.append(data, ignore_index=True)

        self.df.to_excel('Saved/AllTrials.xlsx')

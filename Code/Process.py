from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import numpy as np
import time
import FeatExtract, Crop

class machineLearning():

    def __init__(self, model, selection, extraction, extraction_params, crop=True):
        self.model = model['solver']
        self.crop = crop
        self.name = {'classifier': model['name']}
        self.t ={}

        if selection is not None:
            self.selection = selection['solver']
            self.name['selection'] = selection['name']
        else:
            self.selection = None
            self.name['selection'] = None

        if extraction is not None:
            self.extraction = FeatExtract.FeatureExtraction(extraction, extraction_params)
            self.name['extraction'] = extraction_params['name']
        else:
            self.extraction = None
            self.name['extraction'] = None


    def process(self, imgs, lbls, splits):
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=11)

        for train_index, test_index in skf.split(imgs, lbls):
            x_train, x_test = imgs[train_index], imgs[test_index]
            y_train, y_test = lbls[train_index], lbls[test_index]

            self.t['Start Train'] = time.time()
            self.crop(x_train)
            self.t['Crop Train'] = time.time()
            self.extract()

    def crop(self, imgs):
        if self.crop:
            for i in range(imgs.shape[0]):
                imgs[i] = Crop.croptoface(imgs[i])
        return imgs
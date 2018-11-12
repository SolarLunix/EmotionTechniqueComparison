from databaseForms import JAFFE, KDEF
from FeatureExtraction import HOG, HOG_Parameters, LBP, LBP_Parameters, Gabor, Gabor_Parameters
from MachineLearning import SVM, MLP
from FeatureSelection import PCA, LDA

import numpy as np
import ImageFind
import FeatExtract
import Learn
import save
import time

t = {}
t["Zero"] = time.time()

# This is where you change everything
database = JAFFE
img_size = (130, 130)

extract_params = HOG_Parameters.set_1
extract = HOG
extract_name = extract.name + "1"

name = database.name + extract.name + "1"

select = LDA.Set_1
select_name = LDA.name + "1"

model = SVM.Set_1
model_name = SVM.name + "1"
folds = 10
# DO NOT CHANGE ANYTHING BELOW THIS LINE!

t["Start"] = time.time()
# Assign the Database and return the images and their class as x and y
print('Loading database: {0}'.format(database.name))
imFin = ImageFind.ImageFinder(database.loc, database.form, img_size)
x, y = imFin.returnClasses()
print('\tDatabase Info: \n\t\tLength: {0} \n\t\tNumber of Classes: {1}'.format(len(x), len(database.form)))
t["Read"] = time.time()

featExtract = FeatExtract.FeatureExtraction(extract, extract_params)
x = featExtract.extract(x)
t["Extract"] = time.time()

### Example of saving file
print('\n-- Saving to file: {0}'.format(name))
s = save.SaveFeatures()
s.save(name, x, y)

print('-- Loading file: {0}'.format(name))
tX, tY = s.load(name)
### End of example
t["Load"] = time.time()

ml = Learn.machineLearning(model, select)

acc, times = ml.crossVal(tX, tY, folds)
t["Extract per fold"] = np.array(times["Extract"]).mean()
t["Learning per fold"] = np.array(times["Learn"]).mean()
t["Prediction per fold"] = np.array(times["Predict"]).mean()
t["End"] = time.time()

print("\nLong Description:")
print('\tFeature Extraction: {0} - {1}'.format(extract_name, extract_params))
print('\tFeature Selection: {0} - {1}'.format(select_name, select))
print('\tModel: {0} - {1}'.format(model_name, model))

print("\nShort Description:")
print("\tFeature Extraction:\t\t", extract_name)
print("\tFeature Reduction:\t\t", select_name)
print("\tFeature Classification \t", model_name, "with", folds, "folds")

print('\nAccuracy: {0:.4f}%'.format(acc*100))

print("\nTimes:")
print("\t {0:.5f} \tRead In Time".format(t["Read"] - t["Start"]))
print("\t {0:.5f} \tExtraction Time".format(t["Extract"] - t["Read"]))
print("\t {0:.5f} \tTotal Learning Time".format(t["End"] - t["Load"]))
print("\t------------------------------------------")
print("\t {0:.5f} \tExtraction Time per Fold".format(t["Extract per fold"]))
print("\t {0:.5f} \tLearning Time per Fold".format(t["Learning per fold"]))
print("\t {0:.5f} \tPrediction Time per Fold".format(t["Prediction per fold"]))
print("\t------------------------------------------")
print("\t {0:.5f} \tTotal Time".format(t["End"] - t["Start"]))

names = {
    'extraction': extract_name,
    'db': database.name,
    'selection': select_name,
    'classification': model_name
}

ttimes = {
    'Total': t["End"] - t["Start"],
    'Read In': t["Read"] - t["Start"],
    'Extract': t["Extract"] - t["Read"],
    'Learn': t["End"] - t["Load"],
    'Extract fold': t["Extract per fold"],
    'Learning fold': t["Learning per fold"],
    'Prediction fold': t["Prediction per fold"]
}

saverun = save.SaveRunInfo()
saverun.savetrial(names, acc, ttimes)

print("Trial Saved!")

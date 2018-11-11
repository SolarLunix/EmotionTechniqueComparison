from databaseForms import JAFFE, KDEF
from FeatureExtraction import HOG, HOG_Parameters
from MachineLearning import SVM

import ImageFind
import FeatExtract
import Learn
import save

# Assign the Database and return the images and their class as x and y
database = KDEF
print('Loading database: {0}'.format(database.name))
imFin = ImageFind.ImageFinder(database.loc,database.form, (100,100))
x, y = imFin.returnClasses()
print('\tDatabase Info: \n\t\tLength: {0} \n\t\tNumber of Classes: {1}'.format(len(x),len(database.form)))

featExtract = FeatExtract.FeatureExtraction(HOG, HOG_Parameters.set_1)
x = featExtract.extract(x)

### Example of saving file
name = 'testFile'
print('\n-- Saving to file: {0}'.format(name))
s = save.saveFunction()
s.save(name, x, y)

print('-- Loading file: {0}'.format(name))
tX, tY = s.load(name)
### End of example


model = SVM.SVC_Set_3
ml = Learn.machineLearning(model)
print('\nModel: {0} - {1}'.format(SVM.name, model))
print('\nAccuracy: {0:.4f}%'.format(ml.crossVal(tX,tY,10)*100))





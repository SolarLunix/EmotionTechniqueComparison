from sklearn.svm import SVC

name = 'SVM'

Set_1 = SVC(C=0.1, kernel='linear', random_state=11)
Set_2 = SVC(C=20.0, kernel='rbf', random_state=11)
Set_3 = SVC(C=20.0, kernel='poly', degree=2, random_state=11)

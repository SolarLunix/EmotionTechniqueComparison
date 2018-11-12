from sklearn.svm import SVC

Set_1 = {'name': 'SVM1', 'solver': SVC(C=0.1, kernel='linear', random_state=11)}
Set_2 = {'name': 'SVM2', 'solver':SVC(C=20.0, kernel='rbf', random_state=11)}
Set_3 = {'name': 'SVM3', 'solver':SVC(C=20.0, kernel='poly', degree=2, random_state=11)}

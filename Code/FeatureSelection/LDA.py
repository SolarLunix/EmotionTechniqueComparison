from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

name = 'LDA'

Set_1 = LDA(solver='svd', shrinkage=None, priors=None, n_components=None,
            store_covariance=False, tol=0.0001)
Set_2 = LDA(solver='eigen', shrinkage=None, priors=None, n_components=None,
            store_covariance=False, tol=0.0001)
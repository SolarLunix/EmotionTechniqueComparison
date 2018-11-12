from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

Set_1 = {
    'name': 'LDA1',
    'solver': LDA(solver='svd', shrinkage=None, priors=None, n_components=None,
                       store_covariance=False, tol=0.0001)
}
Set_2 = {
    'name': 'LDA2',
    'solver': LDA(solver='eigen', shrinkage=None, priors=None, n_components=None,
                  store_covariance=False, tol=0.0001)
}
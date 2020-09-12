from sklearn.linear_model import LogisticRegression


def BASELINELOGISTIC():
    logreg = LogisticRegression(random_state=0, n_jobs=4) 
    return logreg

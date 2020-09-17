#rforest.py

from sklearn.ensemble import RandomForestClassifier



defaults = {
    'random_state': 0,
    'n_jobs': 6,
}



def RANDOMFOREST(**kwargs):

    if len(kwargs) > 0:
        return RandomForestClassifier(**kwargs)
    else:
        return RandomForestClassifier(**defaults)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def create_rf(parameters, n_estimators, n_jobs=-1, random_state=None):
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
    )
    rf_clf.set_params(**parameters)
    return rf_clf


def create_adaboost(random_state=None):
    rf_clf = AdaBoostClassifier(n_estimators=50, random_state=random_state)
    return rf_clf

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def create_rf(parameters, random_state=None):
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        # class_weight="balanced",
        # oob_score=True,
        # suggested by scikit docs as good to prevent over-fitting (seems to work a little...?)
        # pre-pruning
        # min_samples_leaf=5
        # prevent over-fitting
        # turns out don't need this with abstaining if low prob. Brings precision down, recall up.
        # HIGHER THAN MFR!!
        # max_depth=3,
    )
    rf_clf.set_params(**parameters)
    # print(rf_clf.cost_complexity_pruning_path(train_X, train_y))
    return rf_clf


def create_adaboost(random_state=None):
    rf_clf = AdaBoostClassifier(n_estimators=50, random_state=random_state)
    return rf_clf

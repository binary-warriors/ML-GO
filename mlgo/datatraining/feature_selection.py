from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold


'''
    For regression: f_regression, mutual_info_regression
    For classification: chi2, f_classif, mutual_info_classif

'''


def select_k_best(train_features, train_targets, no_of_features=None):
    if no_of_features == 0.0:
        no_of_features = None
    if no_of_features >= train_features.shape[1] or no_of_features is None:
        selected_features = SelectKBest(chi2, k='all').fit_transform(train_features, train_targets)
    else:
        selected_features = SelectKBest(chi2, k=no_of_features).fit_transform(train_features, train_targets)
    return selected_features


'''
    threshold=0.0 means keeps only those features with non-zero variance
'''


def variance_based(train_features, train_targets, threshold=0.0):
    model = VarianceThreshold(threshold=threshold)
    try:
        selected_features = model.fit_transform(train_features, train_targets)
    except:
        model = VarianceThreshold()
        selected_features = model.fit_transform(train_features, train_targets)
    return selected_features

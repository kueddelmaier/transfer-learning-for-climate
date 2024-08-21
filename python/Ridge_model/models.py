from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.metrics import explained_variance_score
import numpy as np
from climate_utils import compute_r2

def fit_ridge(X_tr, X_te, Y_tr, Y_te, alphas):
    
    validation_scores = []


    for alpha in alphas:
        clf = Ridge(alpha=alpha)
        clf.fit(X_tr, Y_tr)

        validation_scores.append(compute_r2(Y_te, clf.predict(X_te)))

    chosen_alpha_id = np.argmax(validation_scores)
    chosen_alpha = alphas[chosen_alpha_id]
    #print('chosen aplha: ', chosen_alpha)


    clf = Ridge(alpha=chosen_alpha)
    clf.fit(X_tr, Y_tr)

    return clf
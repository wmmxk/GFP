import GPy
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge


def kernel_ridge(x_fold_tr, y_fold_tr):
    model = GridSearchCV(KernelRidge(), cv=8,
                         param_grid={"alpha": np.logspace(-10, -5, 10),
                                     "gamma": np.logspace(-12, -9, 10),
                                     "kernel": ['laplacian', 'rbf']},
                         scoring='neg_mean_absolute_error', n_jobs=-1)

    model = model.fit(x_fold_tr, y_fold_tr)
    best_model = model.best_estimator_
    best_model.fit(x_fold_tr, y_fold_tr)

    return best_model


def svr(x_fold_tr, y_fold_tr):
    """
    
    :param x_fold_tr: a 2d numpy array
    :param y_fold_tr: a 2d numpy array; the 2nd dim is 1
    :return: return a trained model, whose hyper-parameters are optimized by cv
    """
    svr = GridSearchCV(SVR(kernel='rbf'), cv=3,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)},
                       scoring='neg_mean_absolute_error', n_jobs=-1)
    svr.fit(x_fold_tr, y_fold_tr.ravel())
    model = svr.best_estimator_
    model.fit(x_fold_tr, y_fold_tr.ravel())

    return model


def gp(x_fold_tr, y_fold_tr):
    num_feature = x_fold_tr.shape[1]
    kernel = GPy.kern.Matern52(num_feature, ARD=True) + GPy.kern.White(num_feature)
    model = GPy.models.GPRegression(x_fold_tr, y_fold_tr, kernel)
    model.optimize(messages=True)

    return model

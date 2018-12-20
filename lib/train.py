import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import lib.models as models
from .config import CHUNK_SIZE
from .config import CONFIG
from .config import FILE_TEST
from .config import FILE_TRAIN
from .config import PATH_DATA
from .config import PATH_FIG
from .config import PATH_MODEL
from .config import PATH_OUT_DATA
from .helper.create_dir import create_dir
from .helper.plot_pred_truth import plot_pred_truth


class Train(object):
    def __init__(self, index_chunk):
        """
        load training and testing data, create a model, run CV and predict on testing data

        index_chunk: which chunk to predict
        :param index_chunk:
        """
        self.index_chunk = index_chunk
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test_pred = None
        self.num_feature = None
        self.model = None
        self.preds, self.truths = [], []
        self._prepare_data()
        self._create_res_folder()

    def _prepare_data(self):
        data = pd.read_csv(os.path.join(PATH_DATA, FILE_TRAIN))
        self.num_feature = data.shape[1]-1
        self.X_train = data.values[:, :self.num_feature]
        self.Y_train = data.values[:, self.num_feature:]
        data_test = pd.read_csv(os.path.join(PATH_DATA, FILE_TEST))
        self.X_test = data_test.values
        if CONFIG.sanity:
            self.X_test = self.X_test[:300, ]
        else:
            start = self.index_chunk * CHUNK_SIZE
            end = min((self.index_chunk + 1) * CHUNK_SIZE, self.X_test.shape[0])
            self.X_test = self.X_test[start:end]
        self.Y_test_pred = 0

    @staticmethod
    def _create_res_folder():
        cv_dir = os.path.join(PATH_OUT_DATA, "CV")
        pred_test_dir = os.path.join(PATH_OUT_DATA, "pred")
        cv_fig_dir = os.path.join(PATH_FIG, "CV")
        for one_dir in [cv_dir, pred_test_dir, cv_fig_dir]:
            create_dir(one_dir)

    def cv(self):
        """
        call method, _cv_ond_fold to train a model on the training set and predict on the validation set;
        then call method, _cv_predict to predict on the testing set.
        Finally call method, _cv_save_res to save and visualize the truth_pred on the whole training set,
            prediction on the testing set,
        :return:
        """

        bins = np.linspace(0, max(self.Y_train), 5)
        y_binned = np.digitize(self.Y_train, bins)
        skf = StratifiedKFold(n_splits=CONFIG.num_fold)
        for i, (index_train, index_valid) in enumerate(skf.split(self.X_train, y_binned)):
            self._cv_one_fold(index_train, index_valid, i)
            pred = self._cv_predict()
            self.Y_test_pred += pred
        self.Y_test_pred /= CONFIG.num_fold
        print("Y_pred dimension:--------------- ", self.Y_test_pred.shape)
        self._cv_save_res()

    def _cv_one_fold(self, index_train, index_valid, fold_id):
        """
        train or load a trained model to predict on the testing fold and testing data

        :param index_train: numpy array of (n,); n is the number of training datapoints
        :param index_valid: numpy array of (m,); m is the number of testing datapoints
        :return:
        """
        x_fold_train, y_fold_train = self.X_train[index_train], self.Y_train[index_train]
        x_fold_valid, y_fold_valid = self.X_train[index_valid], self.Y_train[index_valid]
        train_model = getattr(models, CONFIG.type_model)
        model_name = CONFIG.type_model+"_CV_%s_fold_%s" % (str(CONFIG.num_fold), str(fold_id))
        model_path = os.path.join(PATH_MODEL, model_name + ".sav")
        try:
            print("loading trained model ------------%s\n" % model_path)
            self.model = pickle.load(open(model_path, 'rb'))
        except Exception as error:
            print("error-----------", error.args)
            self.model = train_model(x_fold_train, y_fold_train)
            pickle.dump(self.model, open(model_path, 'wb'))
        # the gp model return mean and variance in a tuple
        pred = self.model.predict(x_fold_valid)
        pred = pred[0].flatten() if isinstance(pred, tuple) else pred.flatten()
        self.preds.append(pred)
        self.truths.append(y_fold_valid.flatten())

    def _cv_predict(self):
        """
        make prediction on the testing set.
        :return:
        """
        pred = self.model.predict(self.X_test)
        pred = pred[0] if isinstance(pred, tuple) else pred
        # if the prediction is of (n,), add a new dimension. Otherwise, np.hstack complains in method, _cv_save_res.
        if len(pred.shape) == 1:
            pred = pred[:, None]
        return pred

    def _cv_save_res(self):
        """
        save the prediction in CV along with the truth.
        save the prediction on the testing set.
        plot the prediction vs truth
        :return:
        """
        res_cv = np.stack((np.concatenate(self.truths),
                           np.concatenate(self.preds)), axis=1)
        print("res_cv shape", res_cv.shape)
        res_cv = pd.DataFrame(res_cv)
        res_cv.columns = ["truth", "prediction"]
        res_cv.to_csv(os.path.join(PATH_OUT_DATA, "CV", CONFIG.type_model+"_.csv"), index=False)
        fig_path = os.path.join(PATH_FIG, "CV", CONFIG.type_model+"_"+str(CONFIG.num_fold)+"fold.pdf")
        plot_pred_truth(res_cv.iloc[:, 0], res_cv.iloc[:, 1], fig_path, CONFIG.sanity)

        test_pred = pd.DataFrame(np.hstack((self.X_test, self.Y_test_pred)))
        pred_file = os.path.join(PATH_OUT_DATA, "pred", CONFIG.type_model+"_pred_chunk_%d.csv"% self.index_chunk)
        test_pred.to_csv(os.path.join(PATH_OUT_DATA, pred_file), index=False)


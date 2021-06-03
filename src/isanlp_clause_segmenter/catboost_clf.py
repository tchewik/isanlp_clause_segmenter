import os
import pickle


class CatBoostClf:
    def __init__(self, model_dir_path):
        self._model = pickle.load(open(os.path.join(model_dir_path, 'catboost_clf.pkl'), 'rb'))
        self.DEFAULT_LABEL = 0

    def predict(self, features):
        if type(features) == int and features == -1:
            return self.DEFAULT_LABEL

        return self._model.predict(features)

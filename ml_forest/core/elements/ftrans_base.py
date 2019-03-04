import warnings

import numpy as np

from copy import deepcopy
from ml_forest.core.elements.identity import Base

__author__ = 'LukeLin'


class FTransform(Base):
    def __init__(self, rise, tuning=False):
        """

        :param rise: int, indicates how many stages (see feature_base) to rise:
                          unsupervised learning: 0
                          supervised learning: 1
                          supervised learning with tuning: 1
        :param tuning: boolean, if tuning is happening in the training folds

        rise and tuning are fixed for a FTransform class, so they are not saved in essentials since type is
        """
        super(FTransform, self).__init__()
        if not isinstance(rise, int):
            raise TypeError('The {} method has a non-integer rise, need to be updated'.format(str(type(self))))
        self.__tuning = tuning
        self.__rise = rise
        self.__essentials = {}
        self.__models = None

    @property
    def rise(self):
        return self.__rise

    @property
    def models(self):
        return deepcopy(self.__models)

    def record_models(self, model_collection):
        if self.__models:
            raise ValueError("The attribute models is non-empty. This method does not allow for updating.")

        if not isinstance(model_collection, dict):
            raise TypeError("The parameter model_collection should be a dictionary.")
        self.__models = model_collection

    @property
    def tuning(self):
        return self.__tuning

    @staticmethod
    def decide_element():
        return "FTransform"

    def transform(self, new_X):
        raise NotImplementedError()


# TODO
class FTransWithPreTrainedModels(FTransform):
    def __init(self):
        super(FTransWithPreTrainedModels, self).__init__(rise=0, tuning=False)


class SklearnModel(FTransform):
    def __init__(self, model_type, rise, tuning, **kwargs):
        """

        :param model_type: sklearn model type
        :param rise: whether go to the next stage or not (see Feature class)
        :param tuning: if tuning is part of the sklearn model
        :param kwargs:
        """
        super(SklearnModel, self).__init__(rise=rise, tuning=tuning)
        self.__model_type = model_type

        params = model_type().get_params()
        self.__essentials = deepcopy(params)
        kwargs = locals()["kwargs"]
        for key in kwargs:
            if key in self.__essentials:
                self.__essentials[key] = kwargs[key]

    def prepare_model(self):
        model = self.__model_type()
        for key in model.get_params():
            if key in self.essentials:
                model.set_params(**{key: self.essentials[key]})

        return model

    def fit_singleton(self, x, y, new_x):
        model = self.prepare_model()

        if len(y.shape) > 1 and y.shape[1]==1:
            y = y.ravel()
        model.fit(x, y)

        if "predict_proba" in self.essentials and self.essentials["predict_proba"]:
            values = model.predict_proba(new_x)
            return model, values
        else:
            values = model.predict(new_x)
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
            return model, values

    def fit_whole(self, x):
        model = self.prepare_model()
        value = model.fit_transform(x)

        return model, value

    def transform(self, new_X):
        raise NotImplementedError()


class SklearnUnsupervised(SklearnModel):
    def __init__(self, model_type, rise=0, **kwargs):
        """

        :param model_type: sklearn model type
        :param rise: for unsupervised learning, rise=0 by default
        :param kwargs:
        """
        super(SklearnUnsupervised, self).__init__(model_type, rise=rise, tuning=False, **kwargs)
        self.__essentials = {}

    def transform(self, new_X):
        if len(self.models) > 0:
            warnings.warn("Most likely an unsupervised method returns only one model. Your has more.")

        model = self.models[(0,)]
        value = model.transform(new_X)

        return value


class SklearnRegressor(SklearnModel):
    def __init__(self, model_type, rise=1, tuning=False, **kwargs):
        """

        :param model_type: sklearn model type
        :param rise: for a supervised learning, rise > 0
        :param tuning: whether tuning is part of the model
        :param kwargs:
        """
        super(SklearnRegressor, self).__init__(model_type, rise=rise, tuning=tuning, **kwargs)
        self.__essentials = {}

    def transform(self, new_X):
        lst = []
        for model in self.models.values():
            lst.append(model.predict(new_X))

        stacked_prediction = np.mean(lst, axis=0)
        return stacked_prediction.reshape(-1, 1)


# TODO: Maybe somehow get the encoding dict to allow non-encoded labels
class SklearnClassifier(SklearnModel):
    def __init__(self, model_type, rise=1, tuning=False, **kwargs):
        """

        :param model_type: sklearn model type
        :param rise: for a supervised learning, rise > 0
        :param tuning: whether tuning is part of the model
        :param kwargs:
        """
        super(SklearnClassifier, self).__init__(model_type, rise=rise, tuning=tuning, **kwargs)
        self.__essentials = {}

    def transform(self, new_X):
        if "predict_proba" in self.essentials and self.essentials["predict_proba"]:
            lst = []
            for model in self.models.values():
                lst.append(model.predict_proba(new_X))

            stacked_prediction = np.mean(lst, axis=0)
            return stacked_prediction
        else:
            """
            The majority vote is implemented with np.argmax and np.bincount, 
            whose usage can be demo below:
            
            >> import numpy as np
            
            >> list(map(
            >>     lambda ary: np.bincount(ary),
            >>     np.array([[0,0,1], [1,1,2], [3,3,0]])
            >> ))
            [array([2, 1]), array([0, 2, 1]), array([1, 0, 0, 2])]

            >> np.apply_along_axis(
            >>     lambda ary: np.argmax(np.bincount(ary)),
            >>     axis=1,
            >>     arr = np.array([[0,0,1], [1,1,2], [3,3,0]])
            >> )
            array([0, 1, 3])
            """
            warnings.warn("Currently only works for label encoded labels.")
            lst = []
            for model in self.models.values():
                lst.append(model.predict(new_X))

            stacked_predictions = np.array(lst).T

            majority_vote = np.apply_along_axis(
                lambda ary: np.argmax(np.bincount(ary)),
                axis=1,
                arr= stacked_predictions
            )

            return majority_vote

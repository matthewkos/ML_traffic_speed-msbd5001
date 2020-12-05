import pandas as pd
import torch
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


class Model_builder(object):
    @classmethod
    def skrandforest(cls):
        return SK_RandomForest

    @classmethod
    def skadaboost(cls):
        return SK_AdaBoost

    @classmethod
    def skgradboost(cls):
        return SK_GradientBoosting

    @classmethod
    def skxgblinear(cls):
        return SK_XGBRegressor_gblinear

    @classmethod
    def skxgbtree(cls):
        return SK_XGBRegressor_gbtree

    @classmethod
    def skxgbdart(cls):
        return SK_XGBRegressor_dart


def get_model(model_name):
    return getattr(Model_builder, model_name)()


class SKModel():
    def __init__(self, model):
        self.model = model

    def __call__(self, x, *args, **kwargs):
        return self.predict(x)

    def train(self, x, y):
        self.model.fit(x, y)
        return self.predict(x)

    def predict(self, x):
        return self.model.predict(x)


class SK_RandomForest(SKModel):
    def __init__(self, n, max_depth=None):
        super().__init__(RandomForestRegressor(n_estimators=n, max_depth=max_depth))


class SK_AdaBoost(SKModel):
    def __init__(self, n, max_depth=None):
        super().__init__(AdaBoostRegressor(n_estimators=n, loss='square'))


class SK_GradientBoosting(SKModel):
    def __init__(self, n, max_depth=None):
        super().__init__(GradientBoostingRegressor(n_estimators=n))


class SK_XGBRegressor_gbtree(SKModel):
    def __init__(self, n, max_depth=None, learning_rate=None, subsample=1):
        super().__init__(
            XGBRegressor(n_estimators=n, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample,
                         objective="reg:squarederror", booster='gbtree'))


class SK_XGBRegressor_gblinear(SKModel):
    def __init__(self, n, max_depth=None, learning_rate=None, subsample=1):
        super().__init__(
            XGBRegressor(n_estimators=n, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample,
                         objective="reg:squarederror", booster='gblinear'))


class SK_XGBRegressor_dart(SKModel):
    def __init__(self, n, max_depth=None, learning_rate=None, subsample=1):
        super().__init__(
            XGBRegressor(n_estimators=n, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample,
                         objective="reg:squarederror", booster='dart'))

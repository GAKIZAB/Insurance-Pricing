import xgboost as xgb
import lightgbm as lgb
import numpy as np
from src.models.base_model import PricingModel

class XGBoostPricingModel(PricingModel):
    def __init__(self, name="XGBoost", params=None):
        super().__init__(name)
        self.params = params

    def train(self, X, y, exposure):
        y_freq = y / exposure
        dtrain = xgb.DMatrix(X, label=y_freq, weight=exposure)
        num_round = self.params.get('n_estimators', 100)
        params = {k: v for k, v in self.params.items() if k != 'n_estimators'}
        self.model = xgb.train(params, dtrain, num_round)
        return self.model

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

class LGBMPricingModel(PricingModel):
    def __init__(self, name="LightGBM", params=None):
        super().__init__(name)
        self.params = params

    def train(self, X, y, exposure):
        y_freq = y / exposure
        train_data = lgb.Dataset(X, label=y_freq, weight=exposure)
        num_round = self.params.get('n_estimators', 100)
        params = {k: v for k, v in self.params.items() if k != 'n_estimators'}
        self.model = lgb.train(params, train_data, num_boost_round=num_round)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

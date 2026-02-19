import os
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')

TARGET = 'ClaimNb'
EXPOSURE = 'Exposure'
N_FOLDS = 5
TEST_FOLD = 5

GLM_PARAMS = {'family': 'Poisson', 'link': 'log'}
XGB_PARAMS = {'objective': 'count:poisson', 'n_estimators': 10, 'random_state': 42}
LGBM_PARAMS = {'objective': 'poisson', 'n_estimators': 10, 'random_state': 42}

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
import src.config as config

logger = logging.getLogger(__name__)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_glm(df):
    dat1 = df.copy()
    dat1["AreaGLM"] = dat1["Area"].astype("category").cat.codes
    dat1["VehPowerGLM"] = np.minimum(dat1["VehPower"], 9).astype("category")
    dat1["VehAgeGLM"] = pd.cut(dat1["VehAge"], bins=[-1, 0, 10, 200], labels=["1", "2", "3"]).astype("category")
    dat1["VehAgeGLM"] = dat1["VehAgeGLM"].cat.set_categories(["2", "1", "3"], ordered=True)
    bins_age = [17, 20, 25, 30, 40, 50, 70, 150]
    labels_age = ["1", "2", "3", "4", "5", "6", "7"]
    dat1["DrivAgeGLM"] = pd.cut(dat1["DrivAge"], bins=bins_age, labels=labels_age, include_lowest=True).astype("category")
    dat1["DrivAgeGLM"] = dat1["DrivAgeGLM"].cat.set_categories(["5", "1", "2", "3", "4", "6", "7"], ordered=True)
    dat1["BonusMalusGLM"] = np.minimum(dat1["BonusMalus"], 150)
    dat1["DensityGLM"] = np.log(dat1["Density"])
    dat1["Region"] = dat1["Region"].astype("category")
    if "R24" in dat1["Region"].cat.categories:
        new_order = ["R24"] + [c for c in dat1["Region"].cat.categories if c != "R24"]
        dat1["Region"] = dat1["Region"].cat.set_categories(new_order, ordered=True)
    return dat1

def preprocess_ml(df):
    dat = df.copy()
    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    for col in categorical_cols:
        le = LabelEncoder()
        dat[col] = le.fit_transform(dat[col].astype(str))
    dat["DensityLog"] = np.log(dat["Density"])
    return dat

def create_folds(df, n_folds=5, seed=42):
    np.random.seed(seed)
    df["fold"] = np.random.choice(range(1, n_folds + 1), size=len(df), replace=True)
    return df

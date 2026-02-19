import pandas as pd
import numpy as np
import logging
import src.config as config
from src.data_preprocessing import load_data, preprocess_glm, preprocess_ml, create_folds
from src.evaluation import evaluate_model
from src.models.glm_model import GLMPricingModel
from src.models.ml_models import XGBoostPricingModel, LGBMPricingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting analysis...")
    try:
        data = load_data(config.DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    df_glm = create_folds(preprocess_glm(data))
    df_ml = preprocess_ml(data)
    df_ml['fold'] = df_glm['fold']
    
    train_glm = df_glm[df_glm['fold'] != config.TEST_FOLD]
    test_glm = df_glm[df_glm['fold'] == config.TEST_FOLD]
    train_ml = df_ml[df_ml['fold'] != config.TEST_FOLD]
    test_ml = df_ml[df_ml['fold'] == config.TEST_FOLD]
    
    avg_freq = train_glm[config.TARGET].sum() / train_glm[config.EXPOSURE].sum()
    null_res = evaluate_model("Null", np.full(len(train_glm), avg_freq), train_glm[config.TARGET], train_glm[config.EXPOSURE], np.full(len(test_glm), avg_freq), test_glm[config.TARGET], test_glm[config.EXPOSURE])
    pd_const = null_res["PDW Test"]
    
    glm_f = f"{config.TARGET} ~ AreaGLM + VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + DensityGLM + Region"
    glm2 = GLMPricingModel("GLM2", glm_f)
    glm2.train(train_glm)
    glm2_res = evaluate_model("GLM2", glm2.get_frequency(train_glm), train_glm[config.TARGET], train_glm[config.EXPOSURE], glm2.get_frequency(test_glm), test_glm[config.TARGET], test_glm[config.EXPOSURE], pd_const, 1.0)
    pd_bench = glm2_res["PDW Test"]
    glm2_res["Improvement Index"] = 100.0

    features = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'DensityLog', 'Region']
    
    xgb = XGBoostPricingModel("XGBoost", config.XGB_PARAMS)
    xgb.train(train_ml[features], train_ml[config.TARGET], train_ml[config.EXPOSURE])
    xgb_res = evaluate_model("XGBoost", xgb.predict(train_ml[features]), train_ml[config.TARGET], train_ml[config.EXPOSURE], xgb.predict(test_ml[features]), test_ml[config.TARGET], test_ml[config.EXPOSURE], pd_const, pd_bench)

    lgbm = LGBMPricingModel("LightGBM", config.LGBM_PARAMS)
    lgbm.train(train_ml[features], train_ml[config.TARGET], train_ml[config.EXPOSURE])
    lgbm_res = evaluate_model("LightGBM", lgbm.predict(train_ml[features]), train_ml[config.TARGET], train_ml[config.EXPOSURE], lgbm.predict(test_ml[features]), test_ml[config.TARGET], test_ml[config.EXPOSURE], pd_const, pd_bench)

    results = pd.DataFrame([null_res, glm2_res, xgb_res, lgbm_res])
    print("\n" + results.to_string(index=False))
    results.to_csv("model_comparison_results.csv", index=False)

if __name__ == "__main__":
    main()

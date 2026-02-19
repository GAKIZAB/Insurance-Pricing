import numpy as np
import pandas as pd

def poison_deviance_weighted(pred, obs, exposure):
    pred = np.clip(pred, 1e-10, None)
    obs_safe = np.where(obs == 0, 1e-10, obs)
    dev = exposure * (pred - obs + obs * np.log(obs_safe / pred))
    return 200 * dev.sum() / exposure.sum()

def claim_frequency(obs, exposure):
    return (obs.sum() / exposure.sum()) * 100

def improvement_index(pd_model, pd_const, pd_benchmark):
    if (pd_benchmark - pd_const) == 0: return 0
    return ((pd_model - pd_const) / (pd_benchmark - pd_const)) * 100

def evaluate_model(name, pred_learn, obs_learn, exp_learn, pred_test, obs_test, exp_test, pd_const=None, pd_benchmark=None):
    results = {
        "Model": name,
        "PDW Learn": poison_deviance_weighted(pred_learn, obs_learn, exp_learn),
        "PDW Test": poison_deviance_weighted(pred_test, obs_test, exp_test),
        "CF Learn": claim_frequency(obs_learn, exp_learn),
        "CF Test": claim_frequency(obs_test, exp_test)
    }
    if pd_const is not None and pd_benchmark is not None:
        results["Improvement Index"] = improvement_index(results["PDW Test"], pd_const, pd_benchmark)
    return results

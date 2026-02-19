# Insurance Pricing - Claim Frequency Modeling

This project aims to implement and compare different models for predicting claim frequency in insurance pricing. 

The models compared are:
- **Null Model**: A simple average constant frequency benchmark.
- **GLM (Poisson)**: Generalized Linear Model using the Poisson distribution with specialized features.
- **XGBoost**: Gradient Boosting Machine using the `count:poisson` objective.
- **LightGBM**: Gradient Boosting Machine using the `poisson` objective.

## Project Structure

- `main.py`: The entry point script that orchestrates the data loading, training, and evaluation.
- `src/`:
    - `config.py`: Central configuration for paths and model hyperparameters.
    - `data_preprocessing.py`: Handles feature engineering for both GLM and Machine Learning models.
    - `evaluation.py`: Implementation of Poisson Deviance Weighted (PDW) and Claim Frequency (CF) metrics.
    - `models/`:
        - `glm_model.py`: Poisson GLM implementation using `statsmodels`.
        - `ml_models.py`: Implementations for XGBoost and LightGBM.
- `data/`: Placeholder for the `data.csv` dataset.
- `logs/`: Directory for application logs.


## Evaluation Metrics

Models are evaluated using:
- **PDW (Poisson Deviance Weighted)**: Measures the fit between actual and predicted frequencies, weighted by exposure.
- **GLM2 Improvement Index**: Measures how much better a model performs relative to the benchmark GLM2, with the Null model as the reference point.
- **Claim Frequency**: Ensures predicted frequencies are aligned with actual averages.

## Author
Bertrand GAKIZA

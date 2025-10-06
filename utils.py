import pandas as pd
from sklearn.base import BaseEstimator
from typing import Callable
import numpy as np
import bambi as bmb


def ts_parents(df:pd.DataFrame, causal_structure:dict) -> pd.DataFrame:
    #-- Adds the causal parents of each time series to a DataFrame
    
    data = df.copy()
    # unique causal parents
    causal_parents = {tuple(x) for v in causal_structure.values() for x in v}
    for parent in causal_parents:
        lag = abs(parent[1])
        data.loc[:,f"X{parent[0]}_lag{lag}"] = data[f'X{parent[0]}'].shift(lag)
    data.dropna(inplace=True)
    return data

def train_regime_models(train_data:pd.DataFrame, 
                        causal_structure:dict, 
                        training_model:Callable[..., BaseEstimator], **model_params) -> dict[str, BaseEstimator]:
#-- train model for each time series based on regime causal structure

    models = {}
    for ts, parents in causal_structure.items():
        X = train_data[[f"X{var}_lag{abs(lag)}" for var, lag in parents]]
        y = train_data[f'{ts}']
        model = training_model(**model_params)
        model.fit(X, y)
        models[f'{ts}'] = model

    return models

def structure_to_formulas(structure):
    formulas = {}
    for target, parents in structure.items():
        terms = []
        for parent in parents:
            p, lag = parent
            terms.append(f"X{p}_lag{abs(lag)}")
        if terms:
            formulas[target] = f"{target} ~ " + " + ".join(terms)
    return formulas

def all_formulas_from_structures(causal_structures):
    all_formulas = {}
    for regime, structure in causal_structures.items():
        all_formulas[regime] = structure_to_formulas(structure)
    return all_formulas


def train_regime_models_bayesian(X:pd.DataFrame, 
                                 all_formulas:dict
                                 ) -> dict[int,str,str]:

    regime_models = {}

    for regime, formulas in all_formulas.items():
        train = X[regime] # train data for this regime
        models = {}
        for target, equation in formulas.items():
            model = bmb.Model(equation, data=train, family="gaussian")
            idata = model.fit(draws=1000, tune=2000, chains=4, inference_method='mcmc', progressbar=False, target_accept=0.95)
            models[target] = {"model": model, "idata": idata}
        regime_models[regime] = models
    
    return regime_models



def classify_regime_bayesian(
    X_test: dict[int, pd.DataFrame], 
    all_formulas: dict, 
    bayesian_models: dict,
    causal_structures: dict
) -> tuple[np.ndarray, dict[int, list[float]]]:
    
    n_samples = X_test[0].shape[0]
    predicted_regimes = []
    regime_sample_errors = {regime: [] for regime in causal_structures}

    for i in range(n_samples):
        sample_errors = []

        for regime, formulas in all_formulas.items():
           
            sample = X_test[regime].iloc[i:i+1]
            regime_error = 0
            for target in formulas.keys():
               
                model = bayesian_models[regime][target]["model"]
                idata = bayesian_models[regime][target]["idata"]
                
                inference_data = model.predict(idata=idata, data=sample, kind="pps", inplace=False)
                posterior = inference_data.posterior_predictive[target] 
                y_pred = posterior.mean(dim=("chain", "draw")).values
                y_true = sample[target].values
                regime_error += np.sum((y_true - y_pred) ** 2)

            sample_errors.append(regime_error)
            regime_sample_errors[regime].append(regime_error)
        
        predicted_regimes.append(int(np.argmin(sample_errors)))

    return np.array(predicted_regimes), regime_sample_errors


def sliding_window_regime_prediction(errors_dict: dict[int, list[float]], window_size: int = 5) -> np.ndarray:
    regimes = sorted(errors_dict.keys())
    n_samples = len(next(iter(errors_dict.values())))
    n_windows = n_samples - window_size + 1

    # Convert errors_dict to 2D NumPy array: shape (n_samples, n_regimes)
    errors_array = np.array([errors_dict[r] for r in regimes]).T  # shape: (n_samples, n_regimes)

    # For each window, compute cumulative error and pick regime with lowest
    predicted_regimes = []
    for start in range(n_windows):
        window_errors = errors_array[start:start + window_size]  # shape: (w, n_regimes)
        cumulative_error = window_errors.sum(axis=0)             # shape: (n_regimes,)
        best_regime = int(np.argmin(cumulative_error))
        predicted_regimes.append(best_regime)

    return np.array(predicted_regimes)



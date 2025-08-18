import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
import xgboost as xgb
import datetime
import logging
import joblib   
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

TEST_SIZE = 0.2
CV_NUM = 5
TARGET_COL = 'log_views'
METRICS = {'r_squared': r2_score , 'mse': mean_squared_error , 'mae': mean_absolute_error  }
PARAM_GRIDS = { 
"linear_regression": {
    "fit_intercept": [True, False],
    "positive": [False],
},
"lasso": {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    "fit_intercept": [True, False],
    "selection": ["cyclic", "random"],
    "max_iter": [1000, 5000, 10000]
},
"ridge": {
    "alpha": [0.01, 0.1, 1, 10, 100, 1000],
    "fit_intercept": [True, False],
    "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]

},
"random_forest": {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", 0.5],  
    "bootstrap": [True, False]
},  
"xgb": {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}}

BASELINE_MODELS = {"linear_regression": LinearRegression(),
                    "lasso": Lasso(alpha= 1.0),
                    "ridge": Ridge(alpha= 1.0),
                    "random_forest": RandomForestRegressor(n_estimators= 100, max_depth= None, random_state= 42, n_jobs= -1 ),
                    "xgb":xgb.XGBRegressor(n_estimators = 300, max_depth = 6, learning_rate = 0.1, subsample = 0.8, 
                                colsample_bytree = 0.8, random_state = 42, n_jobs = -1 )
                                }

def baseline_model_evaluation(X_train: pd.Dataframe , X_test: pd.Dataframe, 
                            y_train: pd.Dataframe, y_test: pd.DataFrame, metric: str, metric_func: Callable) -> pd.DataFrame:
    '''
    Trains and evaluates baseline models 

    Args:
        X_train (pd.Dataframe): Training feature dataframe 
        X_test (pd.Dataframe): Testing feature dataframe
        y_train (pd.Dataframe): Training target dataframe 
        y_test (pd.DataFrame): Testing target dataframe 
        metric (str): Metric to evaluate preformance (R squared, MSE, MAE) 
        metric_func (Callable): The sklearn function associated with the metric selected 
    Returns: 
        pd.Dataframe: MSE results dataframe 
    '''
    metric = metric.lower()
    if metric not in METRICS:
        raise ValueError('Metric must be either R squared, MSE or MAE' )
    
    results = {}
    for name, model in BASELINE_MODELS.items():
        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)
        results[name] = metric_func(y_test, model_pred)


    results_df = pd.DataFrame.from_dict(results, orient='index', columns=[metric])
    return results_df



def model_selection(results_df: pd.DataFrame, metric: str) ->  Tuple[BaseEstimator, Dict[str, Any]]:
    '''
    Select model based on baseline model MSE results. Outputs selected model and parameter grid

    Args:
        results_df (Pd.DataFrame): Dataframe with the models and results
        metric (str): Metric to evaluate preformance (R squared, MSE, MAE)
    Returns:
        BaseEstimator: A scikit-learn model instance. 
        dict: parameter grid for selected model 
    ''' 
    metric = metric.lower()
    if metric not in METRICS:
        raise ValueError('Metric must be either R squared, MSE or MAE' )
    
    model_dict = {"linear_regression": LinearRegression(),"lasso": Lasso(),"ridge": Ridge(),
                    "random_forest": RandomForestRegressor(),"xgb":xgb.XGBRegressor() }
    
    results_df = results_df.sort_values(by= metric)
    
    if metric == "r_squared":      
       best_model = results_df.iloc[-1].name
    
    else:     
        best_model = results_df.iloc[0].name

    selected_model = model_dict[best_model]
    selected_param_grid = PARAM_GRIDS[best_model]

    return selected_model, selected_param_grid 


def run_modeling(training_path: str, test_path: str, metric: str = 'mse') ->  Tuple[BaseEstimator, Dict[str, Any], float]:
    '''
    Uses training and evaluation data to assess and choose among baseline models, cross validate the parameters of the chosen 
    model and compare the results

    Args:
        training_path (str): The file path to the processed training CSV
        test_path (str): The file path to the processed test CSV  
        metric (str): Metric to evaluate preformance (R squared, MSE, MAE) 
    Returns:
        BaseEstimator: The selected and hyperparameter tuned model  
        dict: best parameters of selected models 
        float: The mean squared error of the tuned model 
         

    '''
    metric = metric.lower()
    if metric not in METRICS:
        raise ValueError('Invalid Metric: Metric must be either R squared, MSE or MAE' )    
    
    metric_func =  METRICS[metric]

    # Load training and test data
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path)


    # Seperating X and Y for the training and final evaluation data
    y = train_df[TARGET_COL]
    X = train_df.drop([TARGET_COL], axis = 1 )

    y_eval = test_df[TARGET_COL]
    X_eval = test_df.drop([TARGET_COL], axis = 1 )

    # Creates train/test splits for the target variable and features.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SIZE, random_state= 42)

    # Evaluating baseline models
 
    baseline_results_df = baseline_model_evaluation(X_train, X_test, y_train, y_test, metric, metric_func)
    logging.info(baseline_results_df)
    
    
    # Selecting model according to results 
    selected_model, selected_param_grid = model_selection(baseline_results_df, metric)

    # Hyperparameter tuning for selected model
    grid_search = GridSearchCV(estimator= selected_model, param_grid= selected_param_grid, cv = CV_NUM, n_jobs = -1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    logging.info(best_params)

    # Tuned model evaluation using the evaluation data to find the MSE
    tuned_model = selected_model.set_params(**best_params)
    tuned_model.fit(X, y)
    
    tuned_pred = tuned_model.predict(X_eval)   
    tuned_metric = metric_func(y_eval, tuned_pred)
    looging.info(tuned_metric)

    # Saving tuned model 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(tuned_model,f"../models/xgb_model_{timestamp}.json")
    
    
    return tuned_model, best_params, tuned_metric 

    
    
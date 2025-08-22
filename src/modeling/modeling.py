import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
import xgboost as xgb
import datetime
import logging
import joblib
import json 
from typing import Tuple, Callable, Dict, Any 

from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance   
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

TEST_SIZE = 0.2
CV_NUM = 5
TARGET_VAR = 'log_views'
MODEL_DICT = {"linear_regression": LinearRegression(),"lasso": Lasso(),"ridge": Ridge(),
                    "random_forest": RandomForestRegressor(),"xgb":xgb.XGBRegressor() }

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
    "max_features": [None, 'sqrt', 'log2'],  
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
                                colsample_bytree = 0.8, random_state = 42, n_jobs = 1 )
                                }

def start_modeling(training_path: str, test_path: str, metric: str = 'mse', 
                 metric_dict: dict = METRICS, target_var: str = TARGET_VAR, test_size: float = TEST_SIZE ):
    '''
    Loads and prepares training and evaluation data for modeling 
    Args:
        training_path (str): The file path to the processed training CSV
        test_path (str): The file path to the processed test CSV  
        metric (str): Metric to evaluate preformance (R squared, MSE, MAE)
        metric_dict (dict): Dictionary with evaluation metrics and their functions  
        target_var (str): The target variable for modeling 
        test_size (float): Fraction of data left aside for testing 

    Return: 
    pd.DataFrame: Training feature matrix 
    pd.Series: Training target vector 
    pd.DataFrame: Baseline model training feature matrix
    pd.DataFrame: Baseline model testing feature matrix
    pd.Series: Baseline model training target vector
    pd.Series Baseline model testing target vector 
    pd.Dataframe: Evaluation feature matrix
    pd.Series: Evaluation target vector 
    '''
    metric = metric.lower()
    if metric not in metric_dict:
        raise ValueError('Invalid Metric: Metric must be either R squared, MSE or MAE' )    
    
    # Load training and test data
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path)


    # Seperating X and Y for the training and final evaluation data
    y = train_df[target_var]
    X = train_df.drop([target_var], axis = 1 )

    y_eval = test_df[target_var]
    X_eval = test_df.drop([target_var], axis = 1 )

    
    # Creates train/test splits for the target variable and features.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= 42)

    return X, y, X_train, X_test, y_train, y_test, X_eval, y_eval

def baseline_model_evaluation(X_train: pd.DataFrame , X_test: pd.DataFrame, 
                            y_train: pd.DataFrame, y_test: pd.DataFrame,
                              metric: str, baseline_models: dict = BASELINE_MODELS, 
                              metric_dict: dict = METRICS) -> pd.DataFrame:
    '''
    Trains and evaluates baseline models 

    Args:
        X_train (pd.DataFrame): Training feature dataframe 
        X_test (pd.DataFrame): Testing feature dataframe
        y_train (pd.DataFrame): Training target dataframe 
        y_test (pd.DataFrame): Testing target dataframe 
        metric (str): Metric to evaluate preformance (R squared, MSE, MAE) 
        metric_dict (dict): Dictionary with evaluation metrics and their functions
    Returns: 
        pd.DataFrame: MSE results dataframe 
    '''
    metric = metric.lower()
    if metric not in METRICS:
        raise ValueError('Metric must be either R squared, MSE or MAE' )
    
    metric_func = metric_dict[metric]
    results = {}
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)
        results[name] = metric_func(y_test, model_pred)


    results_df = pd.DataFrame.from_dict(results, orient='index', columns=[metric])
    logging.info(results_df)
    return results_df


def model_selection(results_df: pd.DataFrame, metric: str,
                     param_grids: dict = PARAM_GRIDS, 
                     model_dict: dict = MODEL_DICT, 
                     metric_dict: dict = METRICS  ) ->  Tuple[BaseEstimator, Dict[str, Any], str]:
    '''
    Select model based on baseline model MSE results. Outputs selected model and parameter grid

    Args:
        results_df (Pd.DataFrame): Dataframe with the models and results
        metric (str): Metric to evaluate preformance (R squared, MSE, MAE)
        model_dict: Dictionary of models and their sklearn Classes 
        metric_dict (dict): Dictionary with evaluation metrics and their functions    
    Returns:
        BaseEstimator: A scikit-learn model instance. 
        dict: parameter grid for selected model
        str: Name of the model selected  
    ''' 
    metric = metric.lower()
    if metric not in metric_dict:
        raise ValueError('Metric must be either R squared, MSE or MAE' )
    
    results_df = results_df.sort_values(by= metric)
    
    if metric == "r_squared":      
       best_model = results_df.iloc[-1].name
    
    else:     
        best_model = results_df.iloc[0].name

    selected_model = model_dict[best_model]
    selected_param_grid = param_grids[best_model]

    return best_model, selected_model, selected_param_grid 


def plot_important_features(tuned_model: BaseEstimator, X_eval: pd.DataFrame, y_eval: pd.Series,
                         model_name: str, timestamp: str, show: bool = True):
    '''
    Plots and saves feature importances of inputed model

    Args: 
        tuned_model (BaseEstimator): Tuned ensemble model
        model_name (str): Name of the tuned model
        timestamp (str): The timestamp for when the plot is generated. For future comparison and experimentation
        X_eval (pd.DataFrame): Evaluation Feature Dataframe
        y_eval (pd.Series): Evaluation Target Series
        show (bool): Option to show plot
    '''
    result = permutation_importance(
    estimator=tuned_model,
    X=X_eval, 
    y=y_eval,
    n_repeats=10,             
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs= 1
    
)

    perm_df = pd.DataFrame({
        'feature': X_eval.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)

    plt.figure(figsize= (10,10))
    plt.barh(perm_df['feature'], perm_df['importance_mean'])
    plt.xlabel('Decrease in performance after shuffling')
    plt.title(f'Permutation Importance ({model_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../figures/feature-importance/{model_name}_feature_importance_plot_{timestamp}.png')

    if show:
        plt.show()


def plot_predictions(y_true: pd.Series , model_pred: np.ndarray, timestamp: 
                     str, model_name: str, show: bool = True ):
    '''
    Plots and saves predictions of inputed model against the true values

    Args: 
        y_true (pd.Series): The true target values
        model_pred (np.ndarray): The model predicted target values
        model_name (str): Name of the tuned model
        timestamp (str): The timestamp for when the plot is generated. For future comparison and experimentation
        show (bool): Option to show plot    
    '''
    sns.scatterplot(x = y_true, y = model_pred)
    plt.title(" Model Predictions vs True Plot")
    plt.ylabel("Model Predictions")
    plt.xlabel("True Values")
    plt.savefig(f'../figures/predictions/{model_name}_predictions_plot_{timestamp}.png')

    if show:
        plt.show()


def run_hyperparameter_tuning(selected_model_name: str, 
                              selected_model: BaseEstimator, selected_param_grid: dict,
                             X: pd.DataFrame, y: pd. Series, k_folds: int = CV_NUM):
    '''
    Tunes selected model's hyperparamters and returns tuned model its best parameters

    Arg:
        selected_model(BaseEstimator): A scikit-learn model instance. 
        selected_param_grid(dict): Parameter grid for selected model
        selected_model_name(str): Name of the model selected
        X (pd.DataFrame): Training Feature dataframe 
        y (pd.Series): Training target series 
    Return:
        dict: Best parameters of the parameter grid 
        BaseEstimator: The selected model with its best parameters 

    '''

    grid_search = GridSearchCV(estimator= selected_model, param_grid= selected_param_grid, cv = k_folds, n_jobs = 1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    logging.info(best_params)
    tuned_model = selected_model.set_params(**best_params)
    return tuned_model, best_params


def tuned_model_evaluation(tuned_model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
                     X_eval: pd.DataFrame,  y_eval: pd.Series, metric_dict: dict = METRICS ): 
    '''
    Evaluates the tuned model using various metrics (R squared, MSE, MAE). Also outputs the models test data
    predictions
    Args:
        tuned_model (BaseEstimator): Hyperparameter tuned selected model 
        X_eval (pd.DataFrame): Evaluation feature matrix
        y_eval (pd.Series): Evaluation target vector 
        X (pd.DataFrame): Training feature matrix 
        y (pd.Series): Training target vector
        metric_dict (dict): Dictionary with evaluation metrics and their functions
    Return: 
        dict: Metric values for the model evaluated
        np.ndarray: Test data model predictions  
    '''
    tuned_model.fit(X, y)
    
    tuned_pred = tuned_model.predict(X_eval)   
    
    tuned_metrics = {metric: metric_func(y_eval, tuned_pred) for metric, metric_func in metric_dict}
    logging.info(tuned_metrics)
    return tuned_metrics, tuned_pred


def run_interpretation(tuned_model: BaseEstimator, y_true: pd.Series, 
                      X_eval: pd.DataFrame, timestamp: str, model_name: str, 
                      model_pred: np.ndarray):
    '''
    Generating and Saving Plots

    Args:
        tuned_model (BaseEstimator): Hyperparameter tuned selected model 
        X_eval (pd.DataFrame): Evaluation feature matrix
        y_true (pd.Series): Evaluation target vector
        model_name (str): Name of the tuned model
        model_pred (np.ndarray): The model predicted target values
    ''' 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_important_features(tuned_model =  tuned_model, timestamp = timestamp, model_name = model_name, 
                            y_eval= y_true, X_eval= X_eval)
    
    plot_predictions(y_true = y_true, model_pred = model_pred, model_name = model_name, timestamp = timestamp)


def save_artifacts(baseline_results_df: pd.DataFrame, tuned_model: BaseEstimator, 
                       tuned_metrics: float, best_params: dict, model_name: str, 
                       metric: str):
    '''
    Stores artifacts for experiment tracking

    Args: 
        baseline_results_df (pd.DataFrame): Selected metric result for baseline model evaluation  
        tuned_model (BaseEstimator): Hyperparameter tuned selectd model 
        tuned_metrics (float): Evaluations metrics of the tuned model  
        best_params (dict): Best parameters found through hyperparameter tuning   
    ''' 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    baseline_results_df.to_csv( f"../data-results/baseline_results_{timestamp}.csv" )
    with open( f"../logs/{model_name}_metrics_{timestamp}.json", 'w') as m:
        json.dump(tuned_metrics, m )
 


    # Saving tuned model and best parameters 
    joblib.dump(tuned_model,f"../models/{model_name}_model_{timestamp}.pk1")
    with open(f"../models/{model_name}_best_params_{timestamp}.json", 'w') as p:
        json.dump(best_params,p)


def run_modeling(training_path: str, test_path: str, metric: str = 'mse', 
                 metric_dict: dict = METRICS ):
    '''
    Uses training and evaluation data to assess and choose among baseline models, cross validate the parameters of the chosen 
    model and compare the results

    Args:
        training_path (str): The file path to the processed training CSV
        test_path (str): The file path to the processed test CSV  
        metric (str): Metric to evaluate preformance (R squared, MSE, MAE)       
    '''
    metric = metric.lower()
    if metric not in metric_dict:
        raise ValueError('Invalid Metric: Metric must be either R squared, MSE or MAE' )    
    
    X, y, X_train, X_test, y_train, y_test, X_eval, y_eval = start_modeling(training_path= training_path, 
                                                                            test_path= test_path, 
                                                                            metric = metric)

    # Evaluating baseline models
    baseline_results_df = baseline_model_evaluation(X_train, X_test, y_train, y_test, metric)
    
    # Selecting model according to results 
    selected_model_name, selected_model, selected_param_grid = model_selection(baseline_results_df, metric)

    # Hyperparameter tuning for selected model
    tuned_model, best_params = run_hyperparameter_tuning(selected_model_name = selected_model_name, 
                                                         selected_model = selected_model, 
                                                         selected_param_grid = selected_param_grid, X = X, y= y )

    # Tuned model evaluation using the evaluation data
    tuned_metrics, tuned_pred = tuned_model_evaluation(tuned_model= tuned_model, X= X, y= y, 
                                           X_eval= X_eval, y_eval= y_eval)

    # Generating and saving plots for interpretation 
    run_interpretation(tuned_model = tuned_model, y_true = y_eval, 
                      X_eval = X_eval, model_name = selected_model_name, 
                      model_pred = tuned_pred)

    # Saving experiment artifacts
    save_artifacts(baseline_results_df= baseline_results_df, tuned_model= tuned_model, 
                   tuned_metrics= tuned_metrics, best_params= best_params, model_name= selected_model_name,
                   metric= metric,)  
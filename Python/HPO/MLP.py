## MLP model construction
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
# Tensorflow import
# import tensorflow as tf
# # DiCE imports
# import dice_ml
# from dice_ml import Dice
# from dice_ml.utils import helpers  # helper functions
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression, BayesianRidge, ridge_regression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost
# from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_curve,
    brier_score_loss,
    accuracy_score
)
import shap
import pyreadstat
from sklearn.compose import make_column_selector as selector
import copy
import lightgbm as lgb
from sklearn.model_selection import cross_validate
import optuna


def objective(trial: optuna.Trial) -> float:
    minmaxnorm = False
    SMOTE_Process = False
    SHAP_Analysis = True #False True
    ALE_Analysis = True
    print('minmaxnorm: ', minmaxnorm)
    print('SMOTE_Process: ', SMOTE_Process)
    print('SHAP_Analysis: ', SHAP_Analysis)
    print('ALE_Analysis: ', ALE_Analysis)

    ## 导入数据
    dataset = pd.read_csv("../final_data.csv")
    X = dataset[['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease', 'arthritis', 'digestive_disease', 
                 'sleep_duration', 'smoking', 'drinking', 'BMI_status']]
    Y = dataset['successful_aging']


    # 明确指定类别变量和数值变量
    categorical_columns = ['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease', 
                       'arthritis', 'digestive_disease', 'smoking', 'drinking', 'BMI_status']
    numerical_columns = ['sleep_duration']  

    # minmax 处理连续变量，放缩0-1区间，不影响哑变量编码
    numerical_preprocessor = MinMaxScaler()
    dataset[numerical_columns] = numerical_preprocessor.fit_transform(dataset[numerical_columns])

    # pd dummy 处理分类变量，哑变量第一列删除
    cate_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=float)

    datasetX = copy.deepcopy(cate_encoded)
    
    # step 2: Optuna
    data, target = datasetX.values, Y.values
    train_x, x_test, train_y, y_test = train_test_split(data, target, test_size=0.2,
                                                          random_state=23333,shuffle = True)
    
    param = {
        'hidden_layer_sizes': trial.suggest_int("hidden_layer_sizes", 10, 2000),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
        'alpha': trial.suggest_float("alpha", 0.0, 0.1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 
                                                                     'invscaling', 'adaptive']),
        'learning_rate_init': trial.suggest_float("learning_rate_init", 0.000001, 0.1),
        'max_iter': trial.suggest_int("max_iter", 10, 2000),
        'random_state': trial.suggest_int("random_state", 1, 2000),
    }
    
    model = MLPClassifier(**param)  
    
    model.fit(train_x,train_y)
    
    y_pred = model.predict(x_test)
    y_pred_1_proba = model.predict_proba(x_test)[:, 1]
    y_pred_proba = model.predict_proba(x_test)
    tn, fp, fn, tp  = confusion_matrix(y_test, y_pred_proba.argmax(axis=1)).ravel()
    f2 = fbeta_score(y_test, y_pred, average='binary', beta=2)
    f0_5 = fbeta_score(y_test, y_pred, average='binary', beta=0.5)
    f1 = f1_score(y_test, y_pred, average='binary')
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    ROC_AUC = roc_auc_score(y_test, y_pred_1_proba, average='weighted')
    (precisions, recalls, _) = precision_recall_curve(y_test, y_pred_1_proba)
    aucpr = auc(recalls, precisions)
    AP = average_precision_score(y_test, y_pred)
    
    brier = brier_score_loss(y_test, y_pred_1_proba)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
    print('{0} TN: {1} FP: {2} FN: {3} TP: {4} | Pre: {5:.3f} Rec: {6:.3f} F0.5: {7:.3f} F1: {8:.3f} F2: {9:.3f} AP: {10:.3f}| ROC_AUC: {11:.3f} AUCPR: {12:.3f} Brier: {13:.4f} ACC: {14:.4f}'
      .format('Model', tn, fp, fn, tp, precision, recall, f0_5, f1, f2, AP, ROC_AUC, aucpr, brier, acc))
    return ROC_AUC


if __name__ == "__main__":
    
  
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=1000, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, BayesianRidge, ridge_regression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost
from catboost import CatBoostClassifier
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
from alibi.explainers import ALE, plot_ale
from sklearn.compose import make_column_selector as selector
import copy
import lightgbm as lgb

# from imbens.utils._plot import plot_2Dprojection_and_cardinality
from sklearn.model_selection import cross_validate
# np.random.seed(3407)
# random.seed(3407)
# np.random.seed(2333)
# random.seed(2333)
ALE_Analysis = False

dataset = pd.read_csv("../final_data.csv")
X = dataset[['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease', 'arthritis', 'digestive_disease', 
                 'sleep_duration', 'smoking']]
Y = dataset['successful_aging']


# 明确指定类别变量和数值变量
categorical_columns = ['hypertension', 'dyslipidemia', 'liver_disease', 'kidney_disease', 
                       'arthritis', 'digestive_disease', 'smoking']
numerical_columns = ['sleep_duration']  

# minmax 处理连续变量，放缩0-1区间，不影响哑变量编码
numerical_preprocessor = MinMaxScaler()
dataset[numerical_columns] = numerical_preprocessor.fit_transform(dataset[numerical_columns])

# pd dummy 处理分类变量，哑变量第一列删除
cate_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=float)

datasetX = copy.deepcopy(cate_encoded)

Models = {
    'XGB':XGBClassifier(max_depth=9,
    min_child_weight=6,
    gamma=3,
    learning_rate=0.3417813803467016,
    subsample=0.820724210008386,
    colsample_bytree=0.5116227290149833,
    eval_metric='logloss',
    scale_pos_weight=2,
    max_delta_step=2
)
}

x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=23333,
                                                    shuffle = True)

print('Baseline task')
print('Positive in y_test: ', y_test.sum()/y_test.shape[0])
print('Positive in y_train: ',y_train.sum()/y_train.shape[0])

## shap
model_best = Models['XGB']
model_best.fit(x_train, y_train) ## 补充
y_pred_train  = model_best.predict(x_train)
y_pred_train_proba  = model_best.predict_proba(x_train)[:, 1]
Analysis_data = copy.deepcopy(x_train)
Analysis_data['GT'] = y_train
Analysis_data['Pre'] = y_pred_train    
Analysis_data['Proba'] = y_pred_train_proba   
Analysis_True = Analysis_data[Analysis_data['Pre']==Analysis_data['GT']]

def f(x):
        return model_best.predict_proba(x)[:, 1]    
explainer = shap.Explainer(f, x_train)
shap_values = explainer(x_train)

selected_cols = ['Sleep duration', 'Hypertension', 'Arthritis', 'Dyslipidemia', 'Digestive disease', 'Kidney disease',
                 'Smoking', 'Liver disease']

# shap.plots.bar(shap_values, x_train, show=False,
#                       feature_names=selected_cols)
shap.summary_plot(shap_values, x_train, plot_type="bar", show=False,
                      feature_names=selected_cols)
#shap.plots.bar(shap_values.abs.mean(0))
plt.savefig('SHAP_result/Summary_bar_XGB.png', dpi=600, bbox_inches='tight')
plt.show()
#plt.close()

shap.summary_plot(shap_values, x_train, show=False,
                      feature_names=selected_cols, plot_type = "dot")
# shap.plots.beeswarm(shap_values, color=plt.get_cmap("viridis"), feature_names=selected_cols)
plt.savefig('SHAP_result/Summary_ori_XGB.png', dpi=600, bbox_inches='tight')
plt.show()


## 选取tp和fp
# 获取测试集预测
y_pred = Models["XGB"].predict(x_test)
y_prob = Models["XGB"].predict_proba(x_test)[:, 1]

# 转换为 DataFrame 便于索引
X_test_df = pd.DataFrame(x_test, columns=x_train.columns).reset_index(drop=True)
y_test_series = pd.Series(y_test).reset_index(drop=True)

# 四种样本类型
tp = X_test_df[(y_test_series == 1) & (y_pred == 1)]
tn = X_test_df[(y_test_series == 0) & (y_pred == 0)]
fp = X_test_df[(y_test_series == 0) & (y_pred == 1)]
fn = X_test_df[(y_test_series == 1) & (y_pred == 0)]

# 随机抽样各一位个体
selected_instances = {
    "True Positive": tp.sample(1, random_state=1234),
    "True Negative": tn.sample(1, random_state=1234),
    "False Positive": fp.sample(1, random_state=12),
    "False Negative": fn.sample(1, random_state=12)
}

# 选取对应 TP 和 FP 样本进行 SHAP 解释
def get_shap_for_instance(model, X_data, instance):
    shap_explainer = shap.Explainer(model, X_data)
    shap_values = shap_explainer(X_data)
    return shap_values[instance.index[0]]

# 获取 SHAP 值
shap_val_tp = get_shap_for_instance(Models["XGB"], X_test_df, selected_instances["True Positive"])
shap_val_tn = get_shap_for_instance(Models["XGB"], X_test_df, selected_instances["True Negative"])

shap_val_tp.feature_names = selected_cols
shap_val_tn.feature_names = selected_cols

shap.plots.waterfall(shap_val_tp, show=False)
# plt.gcf().set_size_inches(30, 5)
plt.savefig('SHAP_result/Instance_TP_SHAP.png', dpi=600, bbox_inches='tight')
plt.show()
 
shap.plots.waterfall(shap_val_tn, show=False)
# plt.gcf().set_size_inches(25, 5)
plt.savefig('SHAP_result/Instance_TN_SHAP.png', dpi=600, bbox_inches='tight')
plt.show()

## ALE
vals= np.abs(shap_values.values).mean(0)
feature_importance = pd.DataFrame(list(zip(datasetX.columns,vals)),columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
feature_importance.head()
    
explainer = shap.TreeExplainer(model_best)
shap_values = explainer(x_train)
    
shap.summary_plot(shap_values, x_train, plot_type="bar", show=False,
                      feature_names=datasetX.columns)

if ALE_Analysis == True:
    model_ale = ALE(model_best.predict, feature_names=datasetX.columns, 
                    target_names=['successful_aging'])
    model_exp = model_ale.explain(x_train.values) 
    for i in list(feature_importance['col_name'][:7]):
        plot_ale(model_exp, features=[i], n_cols=2, fig_kw={'figwidth':6, 'figheight': 3})
        path = 'SHAP_result/ALE' + i + '.png'
        plt.savefig(path, dpi=330, bbox_inches='tight')

#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
sns.set(style='white', context='notebook', palette='deep')
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


import xgboost as xgb
import lightgbm as lgb
import catboost as cab


# %% Load data
train=pd.read_csv('leave_Office/train.csv')
test = pd.read_csv("leave_Office/test.csv")
# %%
# train['left']=pd.factorize(train['left'])[0]

# %% Join data
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
# %%
dataset.isnull().sum()


#%%
# dataset['fake_id'] = dataset['Work_accident'].map(str) + '_' + dataset['promotion_last_5years'].map(str)  + '_' + dataset['sales'].map(str)  + '_' + dataset['salary'].map(str)
# %% BusinessTravel barplot
g = sns.barplot(x="Work_accident",y="left",data=train)
g = g.set_ylabel("left")
dataset = pd.get_dummies(dataset, columns = ["Work_accident"],prefix="wa", drop_first=True)

# %% average_montly
g = sns.distplot(dataset["average_montly_hours"])
g = g.set_ylabel("left Probability")
# dataset['average_montly_hours'] = np.log1p(dataset['average_montly_hours'])

# %%
g = sns.distplot(dataset["last_evaluation"])
g = g.set_ylabel("left Probability")
# dataset['last_evaluation'] = np.log1p(dataset['last_evaluation'])

# %%
g=sns.barplot(x="number_project",y="left",data=train)
g = g.set_ylabel("left Probability")
# %%
dataset = pd.get_dummies(dataset, columns = ["number_project"],prefix="project", drop_first=True)

# %%
g=sns.barplot(x="promotion_last_5years",y="left",data=train)
g = g.set_ylabel("left Probability")
dataset = pd.get_dummies(dataset, columns = ["promotion_last_5years"],prefix="pl5", drop_first=True)

# %%
g=sns.barplot(x="salary",y="left",data=train)
g = g.set_ylabel("left Probability")
# %%
dataset = pd.get_dummies(dataset, columns = ["salary"],prefix="salary", drop_first=True)

# %%
g=sns.distplot(dataset["satisfaction_level"])
g = g.set_ylabel("left Probability")
# %%
'''
def func(x):
    if x <=0.1:
        return "high"
    elif x <0.4:
        return "low"
    else:
        return 'middle'
dataset['satisfaction_level'] = dataset['satisfaction_level'].apply(func)
'''
dataset['satisfaction_level'] = np.log1p(dataset['satisfaction_level'])
# %%
g=sns.barplot(x="satisfaction_level",y="left",data=train)
g = g.set_ylabel("left Probability")
# %%
# dataset = pd.get_dummies(dataset, columns = ["satisfaction_level"],prefix="satisfaction_level", drop_first=True)

# %%
g=sns.distplot(dataset["time_spend_company"])
g = g.set_ylabel("left Probability")
dataset['time_spend_company'] = np.log1p(dataset['time_spend_company'])

#%%
g=sns.barplot(x="sales",y="left",data=train)
g = g.set_ylabel("left Probability")
#%%
dataset = pd.get_dummies(dataset, columns = ["sales"],prefix="sales", drop_first=True)


# %%
dataset.drop(['id'],axis=1,inplace=True)
# dataset=dataset.drop(["last_evaluation","number_project"],axis=1)






# Modeling
#%% Separate train dataset and test dataset
tr=dataset[:train_len]
ts=dataset[train_len:]
ts.drop(labels=["left"],axis = 1,inplace=True)


#%% Separate train features and label
tr["left"]=tr["left"].astype(int)
Y_train=tr["left"]
X_train=tr.drop(labels = ["left"],axis = 1)
X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())
ts=(ts-ts.min())/(ts.max()-ts.min())

#%%
x_train, x_test, y_train, y_test = train_test_split(X_train,Y_train, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
tpot.export('tpot_attrition_pipeline0211.py')


#%%
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, StandardScaler
from tpot.builtins import StackingEstimator
#exported_pipeline = make_pipeline(
#    StandardScaler(),
#    StackingEstimator(estimator=GaussianNB()),
#    Normalizer(norm="l1"),
#    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.8500000000000001, min_samples_leaf=1, min_samples_split=8, n_estimators=100)
# )
exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.3, min_samples_leaf=1, min_samples_split=3, n_estimators=100)

exported_pipeline.fit(X_train, Y_train)
results = exported_pipeline.predict(ts)




#%% Simple modeling
def ModelAlg(k,r,X_train,Y_train):
    kfold = StratifiedKFold(n_splits=k)
    random_state = r
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state = random_state))
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(cab.CatBoostClassifier())
    classifiers.append(lgb.LGBMClassifier())

    cv_results = []
    for classifier in classifiers :
        cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=1))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis","lightgbm"]})
    
    sns.set(style='white', context='notebook', palette='deep')
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    
    return cv_res

cv_res=ModelAlg(12,2,X_train,Y_train)

cv_res


#%%
kfold = StratifiedKFold(n_splits=10)

#%% RandomForest
RFC = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,Y_train)
RFC_best = gsRFC.best_estimator_
# Best score
gsRFC.best_score_

#%% ExtraTrees
ETC = ExtraTreesClassifier()
## Search grid for optimal parameters
et_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsETC = GridSearchCV(ETC,param_grid = et_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsETC.fit(X_train,Y_train)
ETC_best = gsETC.best_estimator_
# Best score
gsETC.best_score_

#%% AdaBoost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(X_train,Y_train)
ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_


#%%
LGB=lgb.LGBMClassifier()
lgb_params = {'boosting_type': 'gbdt','objective': 'binary','learning_rate' : 0.05, 'verbose': 1,
              'early_stopping_rounds':200,'num_leaves':64,'max_depth':17, 'metric':{'auc'},'nthread': -1}
gslgbDtC=GridSearchCV(LGB,param_grid = lgb_params, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gslgbDtc.fit(X_train,Y_train)
lgb_best = gslgbDtc.best_estimator_
gslgbDTC.best_score_


#%%  Voting Models
votingC = VotingClassifier(estimators=[('RFC',RFC_best),('etc',ETC_best),('ada',ada_best)], voting='soft', n_jobs=4)
votingC = votingC.fit(X_train, Y_train)
votingC.score(X_train,Y_train)

# %% Predicting
y1=votingC.predict_proba(ts)
result = pd.DataFrame()
result['id'] = test['id']
result['left'] = pd.DataFrame(y1)[1]
result["left"]=result["left"].apply(lambda x:1 if x >0.5 else 0)
result.to_csv('key.csv',index=None,header=None,encoding="UTF-8")
# %%



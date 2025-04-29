#!/usr/bin/env python
# coding: utf-8

# 
# # Summary:
# 
# in this notebook, we first make the two Chicago and San Diego datasets similar in terms of features and then we run the baseline ML methods, Transfer learning and domain adaptation models on them. We aim to transfer from one city to another.
# 
# - Classification
# - Timeline Split
# - Results reported after statistical significance test
# - Nan values: Noisy Mean
# - Augmentation: No Aug

# #Libraries

# In[1]:


# pip install adapt stats


# In[2]:


import numpy as np
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import stats


# In[3]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier as xgbc
from sklearn.metrics import roc_curve
from matplotlib import pyplot


# In[4]:


pd.options.display.max_rows = None


# In[5]:


results_f = {
    'RF': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-FA': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-BW': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-CORAL': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-SA': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-FA': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-BW': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-CORAL': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-SA': {'metric':'f', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
}

results_AUC = {
    'RF': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-FA': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-BW': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-CORAL': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-SA': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-FA': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-BW': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-CORAL': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-SA': {'metric':'AUC', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
}
results_f_std = {
    'RF': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-FA': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-BW': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-CORAL': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-SA': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-FA': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-BW': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-CORAL': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-SA': {'metric':'f_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
}

results_np = {
    'RF': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-FA': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-BW': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-CORAL': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-SA': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-FA': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-BW': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-CORAL': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-SA': {'metric':'np', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
}

results_np_std = {
    'RF': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-FA': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-BW': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-CORAL': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-SA': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-FA': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-BW': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-CORAL': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-SA': {'metric':'np_std', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
}

results_np_P = {
    'RF': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-FA': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-BW': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-CORAL': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-SA': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-FA': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-BW': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-CORAL': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-SA': {'metric':'np_P', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
}

results_np_R = {
    'RF': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-FA': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-BW': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-CORAL': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'RF-SA': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-FA': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-BW': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-CORAL': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
    'LR-SA': {'metric':'np_R', 'chi-chi': 0, 'chi-san': 0, 'san-chi':0, 'san-san':0},
}


# In[6]:


def metrics(y_true, y_pred, roundd = False):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    pa, ra, fa, sa = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    if roundd:
      return np.round(f[0],3), np.round(f[1], 3), np.round(fa, 3)
    return f[0], f[1], fa

def ES(y_true, y_pred, roundd = False):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    pa, ra, fa, sa = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return p[0], p[1], r[0], r[1]

def stid(source_name, target_name):
  return (source_name[0:3] + '-' + target_name[0:3]).lower()


# In[7]:


def get_LR_model(XX, YY):
  lr = LogisticRegression()
  param_grid = {
      'solver': ['newton-cg', 'lbfgs', 'liblinear'],
      'C': [1e-2, 1, 10],
      'max_iter': [500, 1000, 1500]
  }
  grid_search = GridSearchCV(estimator = lr, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 10, scoring='f1_weighted')
  grid_search.fit(XX, YY)
  print(grid_search.best_params_)
  return LogisticRegression(max_iter = grid_search.best_params_['max_iter'], C = grid_search.best_params_['C'], solver = grid_search.best_params_['solver'])



def get_RF_model(XX, YY):
  rf = RandomForestClassifier()
  max_depth = [int(x) for x in np.linspace(10, 110, num = 15)]
  max_depth.append(None)
  random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 400, stop = 2000, num = 25)],
               'max_features': ['sqrt', 'log2', None],
               'max_depth': max_depth,
               'min_samples_split': [2, 5, 10, 15, 20],
               'min_samples_leaf': [1, 2, 3, 4, 7],
               'bootstrap': [True, False]

  }

  rf_random = RandomizedSearchCV(estimator = rf, scoring='f1_weighted',
                                 param_distributions = random_grid, n_iter = 110,
                                 cv = 3, verbose=10, random_state=42, n_jobs = -1)
  rf_random.fit(XX, YY)
  print(rf_random.best_params_)
  return RandomForestClassifier(n_estimators=rf_random.best_params_['n_estimators'],
                               max_features=rf_random.best_params_['max_features'],
                               max_depth=rf_random.best_params_['max_depth'],
                               min_samples_split=rf_random.best_params_['min_samples_split'],
                               min_samples_leaf=rf_random.best_params_['min_samples_leaf'],
                               bootstrap=rf_random.best_params_['bootstrap']), rf_random.best_params_

def get_RF(abcd):
    return RandomForestClassifier(n_estimators=abcd['n_estimators'],
                               max_features=abcd['max_features'],
                               max_depth=abcd['max_depth'],
                               min_samples_split=abcd['min_samples_split'],
                               min_samples_leaf=abcd['min_samples_leaf'],
                               bootstrap=abcd['bootstrap'])

def get_RF10(abcd):
    return xgbc(max_depth = abcd['max_depth'],
                               min_child_weight=abcd['min_child_weight'],
                               gamma=abcd['gamma'],
                               subsample=abcd['subsample'],
                               colsample_bytree=abcd['colsample_bytree'],
                               learning_rate=abcd['learning_rate'],
                               n_estimators=abcd['n_estimators'],
                               reg_alpha=abcd['reg_alpha'])
def get_RF_model10(XX, YY):
  xx = xgbc()
  param_dist = {
    'max_depth': [int(x) for x in np.linspace(1, 30, num = 2)],
    'min_child_weight':range(1,20,3),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/100.0 for i in range(50,100,5)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'learning_rate': [1e-5,1e-4,1e-3, 1e-2, 0.1,0.5, 0.9],
    'n_estimators': [2,5,20,80,200,500,800,1200,1700],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    }

  rf_random = RandomizedSearchCV(estimator = xx, scoring='f1_weighted',
                                 param_distributions = param_dist, n_iter = 50,
                                 cv = 3, verbose=2, random_state=42, n_jobs = -1)
  rf_random.fit(XX, YY)
  print(rf_random.best_params_)
  return xgbc(max_depth = rf_random.best_params_['max_depth'],
                               min_child_weight=rf_random.best_params_['min_child_weight'],
                               gamma=rf_random.best_params_['gamma'],
                               subsample=rf_random.best_params_['subsample'],
                               colsample_bytree=rf_random.best_params_['colsample_bytree'],
                               learning_rate=rf_random.best_params_['learning_rate'],
                               n_estimators=rf_random.best_params_['n_estimators'],
                               reg_alpha=rf_random.best_params_['reg_alpha']), rf_random.best_params_


# In[8]:


def stat_significance_trainer_tester(model, X_train, y_train, X_test, y_test, target_X, target_y, source_name, target_name, model_name, plotter = False, reporter =True):

    source_pf, target_pf = [], []
    source_nf, target_nf = [], []
    source_f, target_f = [], []
    sourceNP, sourcePP, targetNP, targetPP = [], [], [], []
    sourceNR, sourcePR, targetNR, targetPR = [], [], [], []

    kf = KFold(n_splits=4, random_state=None, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        model.fit(X_train[train_index], y_train[train_index])
        src_model_src_test_pred = model.predict(X_test)
        src_model_target_test_pred = model.predict(target_X)
        fn,fp,f = metrics(y_test, src_model_src_test_pred)
        source_pf.append(fp)
        source_nf.append(fn)
        source_f.append(f)
        fn,fp,f = metrics(target_y, src_model_target_test_pred)
        target_pf.append(fp)
        target_nf.append(fn)
        target_f.append(f)

        np, pp, nr, pr = ES(y_test, src_model_src_test_pred)
        sourceNP.append(np)
        sourcePP.append(pp)
        sourceNR.append(nr)
        sourcePR.append(pr)
        np, pp, nr, pr = ES(target_y, src_model_target_test_pred)
        targetNP.append(np)
        targetPP.append(pp)
        targetNR.append(nr)
        targetPR.append(pr)

    model.fit(X_train, y_train)
    src_model_src_test_pred = model.predict(X_test)
    fn,fp,f = metrics(y_test, src_model_src_test_pred)
    source_pf.append(fp)
    source_nf.append(fn)
    source_f.append(f)

    src_model_target_test_pred = model.predict(target_X)
    fn,fp,f = metrics(target_y, src_model_target_test_pred)
    target_pf.append(fp)
    target_nf.append(fn)
    target_f.append(f)

    np, pp, nr, pr = ES(y_test, src_model_src_test_pred)
    sourceNP.append(np)
    sourcePP.append(pp)
    sourceNR.append(nr)
    sourcePR.append(pr)
    np, pp, nr, pr = ES(target_y, src_model_target_test_pred)
    targetNP.append(np)
    targetPP.append(pp)
    targetNR.append(nr)
    targetPR.append(pr)

    results_np[model_name][stid(source_name, source_name)] = str(round(numpy.mean(source_nf),3)) + '/' + str(round(numpy.mean(source_pf),3))
    results_f[model_name][stid(source_name, source_name)] = round(numpy.mean(source_f),3)
    results_np[model_name][stid(source_name, target_name)] = str(round(numpy.mean(target_nf),3)) + '/' + str(round(numpy.mean(target_pf),3))
    results_f[model_name][stid(source_name, target_name)] = round(numpy.mean(target_f),3)

    results_np_std[model_name][stid(source_name, source_name)] = str(round(numpy.std(source_nf),3)) + '/' + str(round(numpy.mean(source_pf),3))
    results_f_std[model_name][stid(source_name, source_name)] = round(numpy.std(source_f),3)
    results_np_std[model_name][stid(source_name, target_name)] = str(round(numpy.std(target_nf),3)) + '/' + str(round(numpy.mean(target_pf),3))
    results_f_std[model_name][stid(source_name, target_name)] = round(numpy.std(target_f),3)

    results_np_P[model_name][stid(source_name, source_name)] = str(round(numpy.mean(sourceNP),3)) + '/' + str(round(numpy.mean(sourcePP),3))
    results_np_R[model_name][stid(source_name, source_name)] = str(round(numpy.mean(sourceNR),3)) + '/' + str(round(numpy.mean(sourcePR),3))
    results_np_P[model_name][stid(source_name, target_name)] = str(round(numpy.mean(targetNP),3)) + '/' + str(round(numpy.mean(targetPP),3))
    results_np_R[model_name][stid(source_name, target_name)] = str(round(numpy.mean(targetNR),3)) + '/' + str(round(numpy.mean(targetPR),3))

    if reporter:
      print('----' + model_name + '----')
      print(source_name + ' Model:')
      print(source_name + ' test data:\t\t ',
            '- Stat Significance Test -/+/w mean (STD)',
            round(numpy.mean(source_nf),3), '('+str(round(numpy.std(source_nf),4))+')',
            round(numpy.mean(source_pf),3), '('+str(round(numpy.std(source_pf),4))+')',
            round(numpy.mean(source_f),3), '('+str(round(numpy.std(source_f),4))+')')

      print(target_name + ' test data (Transfered):',
            '- Stat Significance Test -/+/w mean (STD)',
            round(numpy.mean(target_nf),3), '('+str(round(numpy.std(target_nf),4))+')',
            round(numpy.mean(target_pf),3), '('+str(round(numpy.std(target_pf),4))+')',
            round(numpy.mean(target_f),3), '('+str(round(numpy.std(target_f),4))+')')

      print(round(numpy.mean(sourceNP),3),round(numpy.mean(sourcePP),3),round(numpy.mean(sourceNR),3),
           round(numpy.mean(sourcePR),3),round(numpy.mean(targetNP),3),round(numpy.mean(targetPP),3),
           round(numpy.mean(targetNR),3),round(numpy.mean(targetPR),3))

    return model


# In[9]:


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
import math


# In[10]:


def useSMOTE(X, y):
    k_neighbors = math.ceil(sum(y) * 0.01)
    smote = SMOTE(sampling_strategy=1, k_neighbors=k_neighbors)
    X, y = smote.fit_resample(X, y)
    return X, y

def useBORDER(X, y):
    k_neighbors = math.ceil(sum(y) * 0.01)
    m_neighbors = math.ceil(sum(y) * 0.01)

    if k_neighbors == 0: k_neighbors = 1
    if m_neighbors == 0: m_neighbors = 1

    bordersmote = BorderlineSMOTE(sampling_strategy=1,
                                  k_neighbors=k_neighbors,
                                  m_neighbors=m_neighbors)
    X, y = bordersmote.fit_resample(X, y)
    return X, y

def useADASYB(X, y, _random_state = 42):
    n_neighbors = math.ceil(sum(y) * 0.01)
    adasyn = ADASYN(n_neighbors=n_neighbors, random_state = _random_state)
    X, y = adasyn.fit_resample(X, y)
    return X, y


# In[11]:


def np_ratio(y):
  return Counter(y)[0]/Counter(y)[1]


# In[12]:


def importances_rf(rf, name):
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances_sorted = forest_importances.sort_values(ascending=False)
    fig, ax = plt.subplots()
    forest_importances_sorted.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI (RF)")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(name)

def importances_lr(lr, name):
    importances = np.abs(np.array(lr.coef_[0]))
    lr_importances = pd.Series(importances, index=feature_names)
    # Sort the feature importances
    lr_importances_sorted = lr_importances.sort_values(ascending=False)
    fig, ax = plt.subplots()
    lr_importances_sorted.plot.bar(ax=ax)
    ax.set_title("Feature importances using LR Coefficients")
    ax.set_ylabel("")
    fig.tight_layout()
    plt.savefig(name)


# # Dataset

# making two datasets similar

# In[13]:


chi_data = pd.read_csv('./Data-CHI.csv') #Chicago Dataset

chi_data = chi_data.drop(columns={"Unnamed: 0.1", "Unnamed: 0", 'date', 'beach_area', "dayofyear", 'tide_gtm', 'dtide_1', 'dtide_2', 'PrecipSum6',
       'Precip24', 'solar_noon', 'APD', 'DPD', 'turbidity', 'atemp', 'comment', 'dtemp', 'beach', 'WDIR', 'WSPD'})
chi_data['date'] = pd.to_datetime(chi_data[['year', 'month', 'day']])
chi_data['doy'] = chi_data['date'].dt.dayofyear
chi_data.loc[chi_data['logENT'] <= 0, 'logENT'] = 0
chi_data = chi_data.drop(columns={'date', 'month', 'day'})
chi_data['tide_gtm'] = np.where(chi_data['tide'] > chi_data.tide.mean(), 1, 0)

desc = chi_data.describe()
desc.loc['missing'] = [round(i, 3) for i in (10014-np.array(desc)[0])/10014*100]
desc


# In[14]:


san_data = pd.read_csv('./Data-SD.csv') #SanDiego Dataset
san_data['ActivityStartDate'] = pd.to_datetime(san_data['ActivityStartDate'])
san_data['year'] = san_data['ActivityStartDate'].dt.year
san_data = san_data.drop(columns={'visibility', 'MonitoringLocationIdentifier', 'ProjectIdentifier', 'tide_mean',\
                                  'LatitudeMeasure', 'LongitudeMeasure', 'beach_angle', 'WindDir', 'SLP',\
                                  'ActivityStartDate', 'hour', 'WindSpd','AirTemp'})

san_data = san_data.rename(columns={"ResultMeasureValue": "ENT", "WTMP": "Wtemp_B", "Water Level": "tide",\
                                   'DNI_1': 'rad', '3T': 'lograin3T', '7T': 'lograin7T'})

desc = san_data.describe()
desc.loc['missing'] = [round(i, 3) for i in (8730-np.array(desc)[0])/10014*100]
desc


# In[15]:


# san_data = pd.read_csv('./Data-SD-old.csv') #SanDiego Dataset
# beach_list = ['CABEACH_WQX-EH-010', 'CABEACH_WQX-EH-030', 'CABEACH_WQX-EH-200', 'CABEACH_WQX-EH-310', 'CABEACH_WQX-FM-010', 'CABEACH_WQX-IB-010', 'CABEACH_WQX-IB-020', 'CABEACH_WQX-IB-030', 'CABEACH_WQX-IB-040', 'CABEACH_WQX-IB-050', 'CABEACH_WQX-IB-060', 'CABEACH_WQX-MB-053', 'CABEACH_WQX-MB-060', 'CABEACH_WQX-MB-205']
# san_data_filtered = san_data[san_data['beach'].isin(beach_list)] 
# san_data = san_data_filtered
# san_data = san_data.drop(columns={'vo12', 'uo12', 'vo6', 'uo6', 'vo18', 'uo18', 'vo', 'uo','hour', 'wet1', 'lograin1T', 'Sal.', 'beach', "Wind"})
# san_data = san_data.rename(columns={"Temp.": "Wtemp_B"})
# san_data['date'] = pd.to_datetime(san_data[['Year', 'Month', 'Day']])
# san_data['doy'] = san_data['date'].dt.dayofyear
# san_data = san_data.drop(columns={'date', 'Month', 'Day'})
# san_data['tide_gtm'] = np.where(san_data['tide'] > san_data.tide.mean(), 1, 0)
# san_data = san_data.rename(columns={"Year": "year"})

# desc = san_data.describe()
# desc.loc['missing'] = [round(i, 3) for i in (10051-np.array(desc)[0])/10051*100]
# desc


# In[16]:


common_features = [i for i in san_data.columns if i in chi_data.columns]
print(common_features)


# In[17]:


san_data = san_data.loc[:,common_features]
chi_data = chi_data.loc[:,common_features]


# In[18]:


names = ['wet3', 'wet7', 'tide_gtm']
for i in [6,7,10]:
    array1 = np.array(san_data).T[i]
    array2 = np.array(chi_data).T[i]
    print(np_ratio(array1), np_ratio(array2))

feature_names = ['wet3', 'wet7', 'tide_gtm']
ratios_san_diego = [4.405572755417957, 1.726420986883198, 0.6540356195528609]
ratios_chicago = [0.9538633818589026, 0.18284665953218163, 1.8081884464385867]


# In[19]:


names = ['rad', 'tide', 'Wtemp_B', 'lograin3T', 'lograin7T', 'WVHT', 'awind', 'owind']
datata = []
for i in [5,4,3,9,10,2,12,13]:
    array1 = np.array(san_data).T[i]
    array2 = np.array(chi_data).T[i]
    datata.append((array1, array2))
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = axes.flatten()

for i, (array1, array2) in enumerate(datata):
    axes[i].hist(array1, bins=100, alpha=0.5, label='San Diego')
    axes[i].hist(array2, bins=100, alpha=0.5, label='Chicago')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{names[i]} Distributions')
    axes[i].legend()

feature_names = ['wet3', 'wet7', 'tide_gtm']
ratios_san_diego = [4.405572755417957, 1.726420986883198, 0.6540356195528609]
ratios_chicago = [0.9538633818589026, 0.18284665953218163, 1.8081884464385867]
bar_width = 0.25
index = range(len(feature_names))
axes[8].bar(index, ratios_san_diego, bar_width, label='San Diego', color='skyblue')
axes[8].bar([i + bar_width for i in index], ratios_chicago, bar_width, label='Chicago', color='orange')
axes[8].set_xlabel('Features')
axes[8].set_ylabel('0 to 1 Ratios')
axes[8].set_title('Distributions of Categorical Features')
axes[8].set_xticks([i + bar_width / 2 for i in index], feature_names)
axes[8].legend()

plt.tight_layout()
# plt.savefig('dists.png')


# Ready for classification

# In[20]:


chi_threshold = 320
chi_data.loc[chi_data['ENT'] <= chi_threshold, 'ENT'] = 0
chi_data.loc[chi_data['ENT'] > chi_threshold, 'ENT'] = 1


# In[21]:


san_threshold = 30
san_data.loc[san_data['ENT'] <= san_threshold, 'ENT'] = 0
san_data.loc[san_data['ENT'] > san_threshold, 'ENT'] = 1


# drop nan disabled
# 
# cite 1: drop nan, replace with mean, noisy mean

# In[22]:


# chi_data = chi_data.dropna()
# san_data = san_data.dropna()


# In[23]:


feature_names = chi_data.columns[:]
feature_names


# train/test split

# In[24]:


target = 'ENT'


# In[25]:


# for i in range(2014,2023,1):
#     print('year:', i)
#     year_sample = np.array(chi_data.loc[chi_data['year'] == i].loc[:,chi_data.columns == target]).flatten()
#     if len(year_sample):
#         print('\tlen', len(year_sample))
#         print('\tnp ratio', np_ratio(year_sample))


# In[26]:


# for i in range(2014,2023,1):
#     print('year:', i)
#     year_sample = np.array(san_data.loc[san_data['year'] == i].loc[:,san_data.columns == target]).flatten()
#     if len(year_sample):
#         print('\tlen', len(year_sample))
#         print('\tnp ratio', np_ratio(year_sample))


# In[27]:


chi_X_train = np.array(chi_data.loc[chi_data['year'] <= 2018].loc[:,chi_data.columns != target])[:, :-1]
chi_X_test = np.array(chi_data.loc[chi_data['year'] >= 2019].loc[:,chi_data.columns != target])[:, :-1]
chi_y_train = np.array(chi_data.loc[chi_data['year'] <= 2018].loc[:,chi_data.columns == target])
chi_y_test = np.array(chi_data.loc[chi_data['year'] >= 2019].loc[:,chi_data.columns == target])

san_X_train = np.array(san_data.loc[san_data['year'] <= 2018].loc[:,san_data.columns != target])[:, :-1]
san_X_test = np.array(san_data.loc[san_data['year'] == 2019].loc[:,san_data.columns != target])[:, :-1]
san_y_train = np.array(san_data.loc[san_data['year'] <= 2018].loc[:,san_data.columns == target])
san_y_test = np.array(san_data.loc[san_data['year'] == 2019].loc[:,san_data.columns == target])


# In[28]:


print('chi, train <=2018, test >= 2019')
print('sn, train <=2018, test == 2019')


# In[29]:


# target = 'logENT'
# chi_X = np.array(chi_data.loc[:,chi_data.columns != target])
# chi_y = np.array(chi_data[target])
# san_X = np.array(san_data.loc[:,san_data.columns != target])
# san_y = np.array(san_data[target])

# san_X_train, san_X_test, san_y_train, san_y_test = train_test_split(san_X, san_y, test_size=0.2, random_state=42)
# chi_X_train, chi_X_test, chi_y_train, chi_y_test = train_test_split(chi_X, chi_y, test_size=0.2, random_state=42)


# In[30]:


print(chi_X_train.shape)
print(chi_X_test.shape)
print(chi_y_train.shape)
print(chi_y_test.shape)

print(san_X_train.shape)
print(san_X_test.shape)
print(san_y_train.shape)
print(san_y_test.shape)


# replace nan

# In[31]:


chi_train_mean = np.ma.array(chi_X_train, mask=np.isnan(chi_X_train)).mean(axis=0)
san_train_mean = np.ma.array(san_X_train, mask=np.isnan(san_X_train)).mean(axis=0)

chi_train_std = np.ma.array(chi_X_train, mask=np.isnan(chi_X_train)).std(axis=0)
san_train_std = np.ma.array(san_X_train, mask=np.isnan(san_X_train)).std(axis=0)


# In[32]:


# Nan values handling: Noisy mean
for i in range(len(chi_X_test)):
    for j in range(len(chi_X_test[i])):
        if np.isnan(chi_X_test[i][j]):
            chi_X_test[i][j] = np.random.normal(chi_train_mean[j], chi_train_std[j])

for i in range(len(chi_X_train)):
    for j in range(len(chi_X_train[i])):
        if np.isnan(chi_X_train[i][j]):
            chi_X_train[i][j] = np.random.normal(chi_train_mean[j], chi_train_std[j])


# In[33]:


# san_X_test = np.where(np.isnan(san_X_test), san_train_mean, san_X_test)
# san_X_train = np.where(np.isnan(san_X_train), san_train_mean, san_X_train)

# chi_X_test = np.where(np.isnan(chi_X_test), chi_train_mean, chi_X_test)
# chi_X_train = np.where(np.isnan(chi_X_train), chi_train_mean, chi_X_train)


# scaling

# In[34]:


scaler = MinMaxScaler().fit(chi_X_train)
chi_X_train = scaler.transform(chi_X_train)
chi_X_test = scaler.transform(chi_X_test)

scaler = MinMaxScaler().fit(san_X_train)
san_X_train = scaler.transform(san_X_train)
san_X_test = scaler.transform(san_X_test)


# In[35]:


san_y_test = san_y_test.flatten()
san_y_train = san_y_train.flatten()
chi_y_train = chi_y_train.flatten()
chi_y_test = chi_y_test.flatten()


# Aug

# In[36]:


# lax, bur = san_X_train, san_y_train
# ohr, mid = chi_X_train, chi_y_train


# In[37]:


# chi_X_train, chi_y_train = useADASYB(chi_X_train, chi_y_train)
# san_X_train, san_y_train = useADASYB(san_X_train, san_y_train)


# some stats

# In[38]:


print(len(chi_y_test), len(san_y_test), len(chi_y_train), len(san_y_train))


# In[39]:


print('ratios train chi, san', np_ratio(chi_y_train)/(np_ratio(chi_y_train)+1), np_ratio(san_y_train)/(np_ratio(san_y_train)+1))


# In[40]:


print('ratios test chi, san', np_ratio(chi_y_test)/(np_ratio(chi_y_test)+1), np_ratio(san_y_test)/(np_ratio(san_y_test)+1))


# # Models

# In[41]:


chi_regr_rf, chirfparam = get_RF_model(chi_X_train, chi_y_train)
model_RF_chi = stat_significance_trainer_tester(chi_regr_rf, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'RF')


# In[ ]:


san_regr_rf, sanrfparam = get_RF_model(san_X_train, san_y_train)
model_RF_san = stat_significance_trainer_tester(san_regr_rf, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'RF')


# In[ ]:


chi_regr_LR = get_LR_model(chi_X_train, chi_y_train)
model_LR_chi = stat_significance_trainer_tester(chi_regr_LR, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'LR')


# In[ ]:


san_regr_LR = get_LR_model(san_X_train, san_y_train)
model_LR_san = stat_significance_trainer_tester(san_regr_LR, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'LR')


# In[ ]:


#       print(round(numpy.mean(sourceNP),3),round(numpy.mean(sourcePP),3),round(numpy.mean(sourceNR),3),
#        round(numpy.mean(sourcePR),3),round(numpy.mean(targetNP),3),round(numpy.mean(targetPP),3),
#        round(numpy.mean(targetNR),3),round(numpy.mean(targetPR),3))

# importances_rf(model_RF_chi, 'rf-chi.png'), importances_rf(model_RF_san, 'rf-san.png'), importances_lr(model_LR_chi, 'lr-chi.png'), importances_lr(model_LR_san, 'lr-san.png')


# ## ROC

# In[ ]:


#transfering from chicago to san diego
lr_probs = chi_regr_LR.predict_proba(chi_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = chi_regr_rf.predict_proba(chi_X_test)
rf_probs = rf_probs[:, 1]

lr_probs_t = chi_regr_LR.predict_proba(san_X_test)
lr_probs_t = lr_probs_t[:, 1]

rf_probs_t = chi_regr_rf.predict_proba(san_X_test)
rf_probs_t = rf_probs_t[:, 1]

ns_probs = [0 for _ in range(len(chi_y_test))]
ns_auc = roc_auc_score(chi_y_test, ns_probs)
lr_auc = roc_auc_score(chi_y_test, lr_probs)
rf_auc = roc_auc_score(chi_y_test, rf_probs)
lr_auc_t = roc_auc_score(san_y_test, lr_probs_t)
rf_auc_t = roc_auc_score(san_y_test, rf_probs_t)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(chi_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(chi_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(chi_y_test, rf_probs)
lr_fpr_t, lr_tpr_t, _ = roc_curve(san_y_test, lr_probs_t)
rf_fpr_t, rf_tpr_t, _ = roc_curve(san_y_test, rf_probs_t)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR - test: Chicago AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF - test: Chicago AUC=%.3f' % (rf_auc))
pyplot.plot(lr_fpr_t, lr_tpr_t, label='LR - test: San Diego AUC=%.3f' % (lr_auc_t))
pyplot.plot(rf_fpr_t, rf_tpr_t, label='RF - test: San Diego AUC=%.3f' % (rf_auc_t))

results_AUC['LR'][stid("chi", 'chi')] = lr_auc
results_AUC['RF'][stid("chi", 'chi')] = rf_auc
results_AUC['LR'][stid('chi', "san")] = lr_auc_t
results_AUC['RF'][stid('chi', "san")] = rf_auc_t

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('Baseline models - Transfering From Chicago to San Diego')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('transfer-chi.png')


# In[ ]:


#transfering from chicago to san diego
lr_probs = san_regr_LR.predict_proba(san_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = san_regr_rf.predict_proba(san_X_test)
rf_probs = rf_probs[:, 1]

lr_probs_t = san_regr_LR.predict_proba(chi_X_test)
lr_probs_t = lr_probs_t[:, 1]

rf_probs_t = san_regr_rf.predict_proba(chi_X_test)
rf_probs_t = rf_probs_t[:, 1]

ns_probs = [0 for _ in range(len(san_y_test))]
ns_auc = roc_auc_score(san_y_test, ns_probs)
lr_auc = roc_auc_score(san_y_test, lr_probs)
rf_auc = roc_auc_score(san_y_test, rf_probs)
lr_auc_t = roc_auc_score(chi_y_test, lr_probs_t)
rf_auc_t = roc_auc_score(chi_y_test, rf_probs_t)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(san_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(san_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(san_y_test, rf_probs)
lr_fpr_t, lr_tpr_t, _ = roc_curve(chi_y_test, lr_probs_t)
rf_fpr_t, rf_tpr_t, _ = roc_curve(chi_y_test, rf_probs_t)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR - test: San Diego AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF - test: San Diego AUC=%.3f' % (rf_auc))
pyplot.plot(lr_fpr_t, lr_tpr_t, label='LR - test: Chicago AUC=%.3f' % (lr_auc_t))
pyplot.plot(rf_fpr_t, rf_tpr_t, label='RF - test: Chicago AUC=%.3f' % (rf_auc_t))

results_AUC['LR'][stid("san", 'san')] = lr_auc
results_AUC['RF'][stid("san", 'san')] = rf_auc
results_AUC['LR'][stid('san', "chi")] = lr_auc_t
results_AUC['RF'][stid('san', "chi")] = rf_auc_t



# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('Baseline models - Transfering From San Diego to Chicago')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('transfer-san.png')


# # Domain Adaptation

# ## import

# In[ ]:


from adapt.feature_based import FA
from adapt.instance_based import BalancedWeighting
from adapt.feature_based import CORAL
from adapt.feature_based import SA


# In[ ]:


def balanced_subset(X, y, subset_size):
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    positive_subset_indices = np.random.choice(positive_indices, size=subset_size // 2, replace=False)
    negative_subset_indices = np.random.choice(negative_indices, size=subset_size // 2, replace=False)
    subset_indices = np.concatenate([positive_subset_indices, negative_subset_indices])
    np.random.shuffle(subset_indices)
    X_subset = X[subset_indices]
    y_subset = y[subset_indices]
    return np.array(X_subset), np.array(y_subset)


# In[ ]:


# targ_labeled_X_san, targ_labeled_y_san = balanced_subset(lax, bur, 100)
# targ_labeled_X_chi, targ_labeled_y_chi = balanced_subset(ohr, mid, 100)

__, targ_labeled_X_san, __, targ_labeled_y_san = train_test_split(san_X_train, san_y_train, test_size = 0.1196)
__, targ_labeled_X_chi, __, targ_labeled_y_chi = train_test_split(chi_X_train, chi_y_train, test_size = 0.1364)


# In[ ]:


targ_labeled_X_san.shape, targ_labeled_X_chi.shape


# ## FA

# RF

# In[ ]:


crf = get_RF(chirfparam)
chi_rf_fa = FA(crf, Xt=targ_labeled_X_san, yt=targ_labeled_y_san, random_state=0, verbose = 0)
chi_rf_fa = stat_significance_trainer_tester(chi_rf_fa, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'RF-FA')

srf = get_RF(sanrfparam)
san_rf_fa = FA(srf, Xt=targ_labeled_X_chi, yt=targ_labeled_y_chi, random_state=0, verbose = 0)
san_rf_fa = stat_significance_trainer_tester(san_rf_fa, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'RF-FA')


# LR

# In[ ]:


clf = get_LR_model(chi_X_train, chi_y_train)
chi_lr_fa = FA(clf, Xt=targ_labeled_X_san, yt=targ_labeled_y_san, random_state=42, verbose = 0)
chi_lr_fa = stat_significance_trainer_tester(chi_lr_fa, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'LR-FA')

slf = get_LR_model(san_X_train, san_y_train)
san_lr_fa = FA(slf, Xt=targ_labeled_X_chi, yt=targ_labeled_y_chi, random_state=42, verbose = 0)
san_lr_fa = stat_significance_trainer_tester(san_lr_fa, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'LR-FA')


# In[ ]:


#transfering from chicago to san diego
lr_probs = chi_lr_fa.predict_prob(san_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = chi_rf_fa.predict_prob(san_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(san_y_test))]
ns_auc = roc_auc_score(san_y_test, ns_probs)

lr_auc = roc_auc_score(san_y_test, lr_probs)
rf_auc = roc_auc_score(san_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(san_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(san_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(san_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+FA AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+FA AUC=%.3f' % (rf_auc))

results_AUC['LR-FA'][stid("chi", 'san')] = lr_auc
results_AUC['RF-FA'][stid("chi", 'san')] = rf_auc




# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('FA - Transfering From Chicago to San Diego')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('FA-chi.png')


# In[ ]:


#transfering from san to chi
lr_probs = san_lr_fa.predict_prob(chi_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = san_rf_fa.predict_prob(chi_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(chi_y_test))]
ns_auc = roc_auc_score(chi_y_test, ns_probs)

lr_auc = roc_auc_score(chi_y_test, lr_probs)
rf_auc = roc_auc_score(chi_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(chi_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(chi_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(chi_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+FA AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+FA AUC=%.3f' % (rf_auc))

results_AUC['LR-FA'][stid("san", 'chi')] = lr_auc
results_AUC['RF-FA'][stid("san", 'chi')] = rf_auc

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('FA - Transfering From San Diego to Chicago')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('FA-san.png')


# ## BW

# RF

# In[ ]:


regr = get_RF(chirfparam)
crb = BalancedWeighting(regr, Xt=targ_labeled_X_san, yt=targ_labeled_y_san, random_state=0, verbose = 0)
crb = stat_significance_trainer_tester(crb, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'RF-BW')

regr =  get_RF(sanrfparam)
srb = BalancedWeighting(regr, Xt=targ_labeled_X_chi, yt=targ_labeled_y_chi, random_state=0, verbose = 0)
srb = stat_significance_trainer_tester(srb, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'RF-BW')


# LR

# In[ ]:


regr = get_LR_model(chi_X_train, chi_y_train)
clb = BalancedWeighting(regr, Xt=targ_labeled_X_san, yt=targ_labeled_y_san, random_state=0, verbose = 0)
clb = stat_significance_trainer_tester(clb, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'LR-BW')

regr = get_LR_model(san_X_train, san_y_train)
slb = BalancedWeighting(regr, Xt=targ_labeled_X_chi, yt=targ_labeled_y_chi, random_state=0, verbose = 0)
slb = stat_significance_trainer_tester(slb, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'LR-BW')


# In[ ]:


#transfering from chicago to san diego
lr_probs = clb.predict_prob(san_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = crb.predict_prob(san_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(san_y_test))]
ns_auc = roc_auc_score(san_y_test, ns_probs)

lr_auc = roc_auc_score(san_y_test, lr_probs)
rf_auc = roc_auc_score(san_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(san_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(san_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(san_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+BW AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+BW AUC=%.3f' % (rf_auc))

results_AUC['LR-BW'][stid("chi", 'san')] = lr_auc
results_AUC['RF-BW'][stid("chi", 'san')] = rf_auc

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('BW - Transfering From Chicago to San Diego')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('BW-chi.png')


# In[ ]:


#transfering from san to chi
lr_probs = slb.predict_prob(chi_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = srb.predict_prob(chi_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(chi_y_test))]
ns_auc = roc_auc_score(chi_y_test, ns_probs)

lr_auc = roc_auc_score(chi_y_test, lr_probs)
rf_auc = roc_auc_score(chi_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(chi_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(chi_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(chi_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+BW AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+BW AUC=%.3f' % (rf_auc))

results_AUC['LR-BW'][stid("san", 'chi')] = lr_auc
results_AUC['RF-BW'][stid("san", 'chi')] = rf_auc

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('BW - Transfering From San Diego to Chicago')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('BW-san.png')


# ## CORAL

# RF

# In[ ]:


regr = get_RF(chirfparam)
crc = CORAL(regr, Xt=san_X_train, random_state=0, verbose = 0)
crc = stat_significance_trainer_tester(crc, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'RF-CORAL')

regr = get_RF(sanrfparam)
src = CORAL(regr, Xt=chi_X_train, random_state=0, verbose = 0)
src = stat_significance_trainer_tester(src, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'RF-CORAL')


# LR

# In[ ]:


regr = get_LR_model(chi_X_train, chi_y_train)
clc = CORAL(regr, Xt=san_X_train, random_state=0, verbose = 0)
clc = stat_significance_trainer_tester(clc, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'LR-CORAL')

regr = get_LR_model(san_X_train, san_y_train)
slc = CORAL(regr, Xt=chi_X_train, random_state=0, verbose = 0)
slc = stat_significance_trainer_tester(slc, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'LR-CORAL')


# In[ ]:


#transfering from chicago to san diego
lr_probs = clc.predict_prob(san_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = crc.predict_prob(san_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(san_y_test))]
ns_auc = roc_auc_score(san_y_test, ns_probs)

lr_auc = roc_auc_score(san_y_test, lr_probs)
rf_auc = roc_auc_score(san_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(san_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(san_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(san_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+CORAL AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+CORAL AUC=%.3f' % (rf_auc))

results_AUC['LR-CORAL'][stid("chi", 'san')] = lr_auc
results_AUC['RF-CORAL'][stid("chi", 'san')] = rf_auc

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('CORAL - Transfering From Chicago to San Diego')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('c-chi.png')


# In[ ]:


#transfering from san to chi
lr_probs = slc.predict_prob(chi_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = src.predict_prob(chi_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(chi_y_test))]
ns_auc = roc_auc_score(chi_y_test, ns_probs)

lr_auc = roc_auc_score(chi_y_test, lr_probs)
rf_auc = roc_auc_score(chi_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(chi_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(chi_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(chi_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+CORAL AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+CORAL AUC=%.3f' % (rf_auc))

results_AUC['LR-CORAL'][stid("san", 'chi')] = lr_auc
results_AUC['RF-CORAL'][stid("san", 'chi')] = rf_auc

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('CORAL - Transfering From San Diego to Chicago')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('c-san.png')


# ## SA

# RF

# In[ ]:


regr = get_RF(chirfparam)
crs = SA(regr, Xt=san_X_train, random_state=0, verbose = 0)
crs = stat_significance_trainer_tester(crs, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'RF-SA')

regr = get_RF(sanrfparam)
srs = SA(regr, Xt=chi_X_train, random_state=0, verbose = 0)
srs = stat_significance_trainer_tester(srs, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'RF-SA')


# LR

# In[ ]:


regr = get_LR_model(chi_X_train, chi_y_train)
cls = SA(regr, Xt=san_X_train, random_state=0, verbose = 0)
cls = stat_significance_trainer_tester(cls, chi_X_train, chi_y_train, chi_X_test, chi_y_test, san_X_test, san_y_test, source_name = "Chicago", target_name = 'San Diego', model_name = 'LR-SA')

regr = get_LR_model(san_X_train, san_y_train)
sls = SA(regr, Xt=chi_X_train, random_state=0, verbose = 0)
sls = stat_significance_trainer_tester(sls, san_X_train, san_y_train, san_X_test, san_y_test, chi_X_test, chi_y_test, source_name = "San Diego", target_name = 'Chicago', model_name = 'LR-SA')


# In[ ]:


#transfering from chicago to san diego
lr_probs = cls.predict_prob(san_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = crs.predict_prob(san_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(san_y_test))]
ns_auc = roc_auc_score(san_y_test, ns_probs)

lr_auc = roc_auc_score(san_y_test, lr_probs)
rf_auc = roc_auc_score(san_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(san_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(san_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(san_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+SA AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+SA AUC=%.3f' % (rf_auc))

results_AUC['LR-SA'][stid("chi", 'san')] = lr_auc
results_AUC['RF-SA'][stid("chi", 'san')] = rf_auc

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('SA - Transfering From Chicago to San Diego')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('s-chi.png')


# In[ ]:


#transfering from san to chi
lr_probs = sls.predict_prob(chi_X_test)
lr_probs = lr_probs[:, 1]

rf_probs = srs.predict_prob(chi_X_test)
rf_probs = rf_probs[:, 1]

ns_probs = [0 for _ in range(len(chi_y_test))]
ns_auc = roc_auc_score(chi_y_test, ns_probs)

lr_auc = roc_auc_score(chi_y_test, lr_probs)
rf_auc = roc_auc_score(chi_y_test, rf_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(chi_y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(chi_y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(chi_y_test, rf_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Classifier AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, label='LR+SA AUC=%.3f' % (lr_auc))
pyplot.plot(rf_fpr, rf_tpr, label='RF+SA AUC=%.3f' % (rf_auc))

results_AUC['LR-SA'][stid("san", 'chi')] = lr_auc
results_AUC['RF-SA'][stid("san", 'chi')] = rf_auc

# axis labels
pyplot.xlabel('1 - Specificity')
pyplot.ylabel('Sensitivity')
pyplot.title('SA - Transfering From San Diego to Chicago')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
# plt.savefig('s-san.png')


# # Results Export

# In[ ]:


results_f_transposed = {key: [value['metric'], value['chi-chi'], value['chi-san'], value['san-chi'], value['san-san']] for key, value in results_f.items()}
results_f = pd.DataFrame.from_dict(results_f_transposed, orient='index', columns=['metric', 'chi-chi', 'chi-san', 'san-chi', 'san-san'])

results_f_std_transposed = {key: [value['metric'], value['chi-chi'], value['chi-san'], value['san-chi'], value['san-san']] for key, value in results_f_std.items()}
results_f_std = pd.DataFrame.from_dict(results_f_std_transposed, orient='index', columns=['metric', 'chi-chi', 'chi-san', 'san-chi', 'san-san'])

results_np_transposed = {key: [value['metric'], value['chi-chi'], value['chi-san'], value['san-chi'], value['san-san']] for key, value in results_np.items()}
results_np = pd.DataFrame.from_dict(results_np_transposed, orient='index', columns=['metric', 'chi-chi', 'chi-san', 'san-chi', 'san-san'])

results_np_std_transposed = {key: [value['metric'], value['chi-chi'], value['chi-san'], value['san-chi'], value['san-san']] for key, value in results_np_std.items()}
results_np_std = pd.DataFrame.from_dict(results_np_std_transposed, orient='index', columns=['metric', 'chi-chi', 'chi-san', 'san-chi', 'san-san'])

results_np_P_transposed = {key: [value['metric'], value['chi-chi'], value['chi-san'], value['san-chi'], value['san-san']] for key, value in results_np_P.items()}
results_np_P = pd.DataFrame.from_dict(results_np_P_transposed, orient='index', columns=['metric', 'chi-chi', 'chi-san', 'san-chi', 'san-san'])

results_np_R_transposed = {key: [value['metric'], value['chi-chi'], value['chi-san'], value['san-chi'], value['san-san']] for key, value in results_np_R.items()}
results_np_R = pd.DataFrame.from_dict(results_np_R_transposed, orient='index', columns=['metric', 'chi-chi', 'chi-san', 'san-chi', 'san-san'])

results_AUC_transposed = {key: [value['metric'], value['chi-chi'], value['chi-san'], value['san-chi'], value['san-san']] for key, value in results_AUC.items()}
results_AUC = pd.DataFrame.from_dict(results_AUC_transposed, orient='index', columns=['metric', 'chi-chi', 'chi-san', 'san-chi', 'san-san'])


res_df = pd.concat([results_f, results_f_std, results_np, results_np_std, results_np_P, results_np_R, results_AUC], keys=['results_f', 'results_f_std', 'results_np', 'results_np_std', 'results_np_P', 'results_np_R', 'results_np_AUC'])


res_df.to_csv('./concatenated_results0.csv')


# test

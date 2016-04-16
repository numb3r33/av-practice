
# coding: utf-8

# ## Loan Prediction

# ** Objectives **
# 
# * Feature Selection

# In[91]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import StratifiedKFold, cross_val_score, LeaveOneOut

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier

get_ipython().magic(u'matplotlib inline')


# In[123]:

train = pd.read_csv('./data/synthesized/train_mean.csv')
test = pd.read_csv('./data/synthesized/test_mean.csv')
sub = pd.read_csv('./data/Sample_Submission_ZAuTl8O.csv')

train_original = pd.read_csv('./data/train_u6lujuX.csv', index_col='Loan_ID')
test_original = pd.read_csv('./data/test_Y3wMUE5.csv')
feature_names = train_original.columns[:-1]


# In[43]:

def get_data(df, dataset_type='train'):
    if dataset_type == 'train':
        features = get_train_features(df)
        X = df[features]
        y = df.Loan_Status
        return X, y
    else:
        features = get_test_features(df)
        X = df[features]
        return X

def get_train_features(df):
    features = df.columns[1:-1]
    return features

def get_test_features(df):
    features = df.columns[1:]
    return features


# In[44]:

X, y = get_data(train, dataset_type='train')
X_test = get_data(test, dataset_type='test')


# In[45]:

assert (X.shape[1] == X_test.shape[1]), 'Mismatch in number of features'


# ## Cross validation

# In[50]:

def skfold_scorer(scoring = 'accuracy', n_folds = 5):
    def score(model, X, y):
        return np.mean(cross_val_score(model, X, y, cv = n_folds, scoring = scoring, n_jobs = -1))
    return score

def loo_scorer(scoring = 'accuracy'):
    def score(model, X, y):
        return np.mean(cross_val_score(model, X, y, cv = LeaveOneOut(X.shape[0]), scoring = scoring, n_jobs = -1))
    return score


# ## Models

# In[48]:

logreg = LogisticRegression()
rf = RandomForestClassifier(n_jobs=-1)
gbm = GradientBoostingClassifier()


# ## Feature Selection

# In[39]:

def exhaustive_search(X_train, y_train, model, scorer, d = 0.1):
    q_max = 0
    n_features = X_train.shape[1] 
    best_features = []
    
    for j in range(1, n_features + 1):
        feature_indices = itertools.combinations(range(n_features), j)
        for features_list in feature_indices:
            sub_data = X_train.iloc[:, features_list]
            q = scorer(model, sub_data, y_train)
            if q > q_max:
                if abs(q - q_max) < d*q_max:
                    q_max = q
                    j_min = j
                    best_features = features_list
                    return best_features, q_max
                else:
                    q_max = q
                    j_min = j
                    best_features = features_list
                    
    return best_features, q_max


# In[108]:

next(itertools.combinations([1, 2, 3], 2))


# In[59]:

def show_results(algorithm, data, target, model, scorer):
    best_features, best_Q = algorithm(data, target, model, scorer)
    print 'Best score = ' + str(best_Q)
    print 'Best features:'  + str(best_features)
    print '----------'


# In[60]:

show_results(exhaustive_search, X, y, logreg, skfold_scorer())


# In[61]:

show_results(exhaustive_search, X, y, rf, skfold_scorer())


# In[62]:

show_results(exhaustive_search, X, y, gbm, skfold_scorer())


# ## Training

# In[96]:

model = XGBClassifier(n_estimators=300)


# In[109]:

features = [X.columns[i] for i in [1, 9, 12, 13]]
features


# In[113]:

X_sub = X[features]
test_sub = test[features]


# In[114]:

assert ( X_sub.shape[1] == test_sub.shape[1] ), 'Mismatch in number of features'


# In[115]:

model.fit(X_sub, y)


# In[119]:

prediction = model.predict(test_sub)


# In[120]:

plt.hist(prediction);


# In[124]:

sub['Loan_ID'] = test_original.Loan_ID
sub['Loan_Status'] = prediction


# In[125]:

sub.to_csv('../submissions/xgb_submission.csv', index=False)


# In[ ]:





# coding: utf-8

# ## Loan Prediction

# ### Objectives:
# * Classify loan prediction by using Logistic Regession
# * Show how Newton Raphson Method or Gradient Descent can be used to optimize the algorithm
# * Plot ROC curves for different splits of the training set

# In[167]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

get_ipython().magic(u'matplotlib inline')


# In[213]:

# Set random seed
np.random.seed(144)


# In[214]:

train = pd.read_csv('./data/train_u6lujuX.csv')
test = pd.read_csv('./data/test_Y3wMUE5.csv')
sub = pd.read_csv('./data/Sample_Submission_ZAuTl8O.csv')


# In[215]:

train.head()


# In[216]:

test.head()


# In[217]:

# check to see if there is an overlap in loan ids among training and test examples
len(set(train.Loan_ID) & set(test.Loan_ID))


# In[218]:

# Set Loan Id as index
train = train.set_index('Loan_ID')
test = test.set_index('Loan_ID')


# In[219]:

test_loan_ids = test.index


# ## One Hot Encoding

# In[220]:

features = train.columns.drop('Loan_Status')

target = train.Loan_Status
train = train[features].T.to_dict().values()

test = test.T.to_dict().values()


# In[221]:

transformer = DictVectorizer(sparse=False)

train = transformer.fit_transform(train)
test = transformer.transform(test)


# In[222]:

# fill missing values with -1
X = pd.DataFrame(train)
test = pd.DataFrame(test)

y = pd.Series(target)


# In[223]:

X = X.fillna(-1)
test = test.fillna(-1)


# In[224]:

y = (y=='Y').astype(np.int)


# ## Cross validation scores

# In[70]:

skf = StratifiedKFold(y.values, n_folds=5, random_state=44)
C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# In[71]:

def get_roc_scores(X, y, cv):
    return [
        cross_val_score(LogisticRegression(C=c), X, y, cv=cv, scoring='roc_auc').mean()
        for c in C_grid
    ]

roc_scores_by_C = get_roc_scores(X, y, skf)


# In[72]:

plt.plot(C_grid, roc_scores_by_C)
plt.xlabel('C')
plt.ylabel('ROC AUC Score')
plt.title('Relationship between C and ROC AUC Score');


# ** Best performing value for C **

# In[73]:

C_grid[np.argmax(roc_scores_by_C)]


# In[74]:

# Now lets scale the input vector and then perform the analysis again
X = scale(X)

roc_scores_by_C = get_roc_scores(X, y, skf)


# In[75]:

plt.plot(C_grid, roc_scores_by_C)
plt.xlabel('C')
plt.ylabel('ROC AUC Score')
plt.title('Relationship between C and ROC AUC Score');


# In[76]:

C_grid[np.argmax(roc_scores_by_C)]


# ** Very stable cv score **

# ## Lets see whether choice of solver effects prediction score or not

# In[150]:

solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag']

def get_roc_scores_solver(X, y, cv):
    return [
        cross_val_score(LogisticRegression(C=0.1, solver=solver), X, y, cv=cv, scoring='roc_auc').mean()
        for solver in solver_list
    ]

roc_scores_by_solver = get_roc_scores_solver(X, y, skf)


# In[155]:

plt.plot(np.arange(0, 4), roc_scores_by_solver)
plt.xticks(np.arange(0, 4), solver_list, rotation='vertical')
plt.xlabel('Solver')
plt.ylabel('ROC AUC Score')
plt.title('Relationship between solver and ROC AUC Score');


# ** For small datasets like this one liblinear is a good choice **

# ## Let's see how roc-auc score varies on different train-test splits

# In[114]:

def get_roc_by_splits(X, y):
    seeds = [44, 123, 279, 512, 1279, 3022]
    clf = LogisticRegression(C=0.1)
    cv_scores = []
    oos_scores = []
    
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.15)
        
        skf = StratifiedKFold(y_train, n_folds=5, random_state=44)
        mean_cv_score = cross_val_score(clf, X_train, y_train, cv=skf, scoring='roc_auc').mean()
        cv_scores.append(mean_cv_score)
        
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        oos_scores.append(roc_auc_score(y_test, preds))
    
    return (seeds, cv_scores, oos_scores)


# In[115]:

seeds, cv_scores, oos_scores = get_roc_by_splits(X, y)


# In[116]:

plt.plot(seeds, cv_scores)
plt.xlabel('Seed value for Train Test Split')
plt.ylabel('Mean CV score (5-fold)')
plt.title('Relationship between different different seed values and cv scores');


# In[117]:

plt.plot(seeds, oos_scores)
plt.xlabel('Seed value for train test split')
plt.ylabel('Out of sample score ')
plt.title('Relationship between different splits and oos score');


# ## Plot ROC Curves

# In[124]:

def get_roc_dfs(X, y):
    seeds = [44, 123, 279, 512, 1279, 3022]
    clf = LogisticRegression(C=0.1)
    
    roc_dfs = []
    
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.15)
        
        clf.fit(X_train, y_train)
        fpr, tpr, _ = roc_curve(y_test, preds)
        
        roc_dfs.append(pd.DataFrame(dict(fpr=fpr, tpr=tpr)))
        
    return roc_dfs


# In[125]:

roc_dfs = get_roc_dfs(X, y)


# In[140]:

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
row_index = 0
col_index = 0

for i in range(len(roc_dfs)):    
    ax[row_index][col_index].plot(roc_dfs[i].fpr, roc_dfs[i].tpr)
    ax[row_index][col_index].plot(ax[row_index][col_index].get_xlim(), ax[row_index][col_index].get_ylim(), ls="--", c=".3")
    ax[row_index][col_index].set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1));

    col_index += 1
    
    if col_index > 1:
        row_index += 1
        col_index = 0
    


# In[225]:

## Prediction
clf = LogisticRegression(C=0.1, solver='liblinear')
clf.fit(X, y)


# In[226]:

predictions = clf.predict(test)


# In[231]:

def encode_labels(prediction):
    if prediction == 1:
        return 'Y'
    else:
        return 'N'
    
predictions_encoded = map(encode_labels, predictions)


# In[234]:

sub.loc[: ,'Loan_ID'] = test_loan_ids
sub.loc[:, 'Loan_Status'] = predictions_encoded


# In[235]:

sub.head()


# In[236]:

sub.to_csv('./submissions/baseline_submission.csv', index=False)


# In[ ]:




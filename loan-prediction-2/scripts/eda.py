# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:12:29 2016

@author: abhishek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer

train = pd.read_csv('./data/train_u6lujuX.csv', index_col='Loan_ID')
test = pd.read_csv('./data/test_Y3wMUE5.csv', index_col='Loan_ID')
sub = pd.read_csv('./data/Sample_Submission_ZAuTl8O.csv')

# log transformation
class Dataset(object):
    
    def __init__(self, train, test, strategy, train_filename, test_filename):
        self.train = train.copy()
        self.test = test.copy()
        self.strategy = strategy
        self.train_filename = train_filename
        self.test_filename = test_filename
        
        self.y = (train.Loan_Status == 'Y').astype(int)
        self.transformer = DictVectorizer(sparse=False)
        
    def most_common_feature(self, feature_name):
        common_feature_train = self.train[feature_name].value_counts().index[0]
        common_feature_test = self.test[feature_name].value_counts().index[0]
        
        return common_feature_train, common_feature_test
    
    def new_features(self):
        total_income = self.train['ApplicantIncome'] + self.train['CoapplicantIncome']
        self.train['EMI'] = total_income / self.train.Loan_Amount_Term
        
        total_income_test = self.test['ApplicantIncome'] + self.test['CoapplicantIncome']
        self.test['EMI'] = total_income_test / self.test.Loan_Amount_Term
        
    def fill_missing_values(self):
        features = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
        
        for feat in features:
            common_val_train, common_val_test = self.most_common_feature(feat)
            self.train[feat] = self.train[feat].fillna(common_val_train)
            self.test[feat] = self.test[feat].fillna(common_val_test)
         
        numerical_features = ['LoanAmount', 'Loan_Amount_Term', 'EMI']
        
        for feat in numerical_features:
            if self.strategy == 'mean':
                self.train[feat] = self.train[feat].fillna(self.train[feat].mean())
                self.test[feat] = self.test[feat].fillna(self.test[feat].mean())
            elif self.strategy == 'median':
                self.train[feat] = self.train[feat].fillna(self.train[feat].median())
                self.test[feat] = self.test[feat].fillna(self.test[feat].median())
            else:
                self.train[feat] = self.train[feat].fillna(self.train[feat].mode())
                self.test[feat] = self.test[feat].fillna(self.test[feat].mode())
    
    def one_hot_encoding(self):
        features = self.train.columns.drop('Loan_Status')
        
        self.X = self.train[features].T.to_dict().values()
        self.X_test = self.test[features].T.to_dict().values()
        
        self.X = self.transformer.fit_transform(self.X)
        self.X_test = self.transformer.transform(self.X_test)
    
    def save_dfs(self):
        self.X = pd.DataFrame(self.X)
        self.X_test = pd.DataFrame(self.X_test)
        
        self.X.loc[:, 'Loan_Status'] = self.y
        
        self.X.to_csv('./data/synthesized/' + self.train_filename, index=False)
        self.X_test.to_csv('./data/synthesized/' + self.test_filename, index=False)
    
    def prepareDataset(self):
        self.new_features()
        self.fill_missing_values()
        self.transformation()
        self.one_hot_encoding()
        self.save_dfs()

        
    
    def transformation(self):
        '''
        Log transformation of features
        '''
        
        self.train['ApplicantIncome'] = self.train.ApplicantIncome.map(np.log1p)
        self.test['ApplicantIncome'] = self.test.ApplicantIncome.map(np.log1p)
        
        self.train['CoapplicantIncome'] = self.train.CoapplicantIncome.map(np.log1p)
        self.test['CoapplicantIncome'] = self.test.CoapplicantIncome.map(np.log1p)
        
        self.train['LoanAmount'] = self.train.LoanAmount.map(np.log1p)
        self.test['LoanAmount'] = self.test.LoanAmount.map(np.log1p)
        
        
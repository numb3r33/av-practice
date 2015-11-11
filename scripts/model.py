from helper import inverse_mapping

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import numpy as np

class Model(object):

    def __init__(self, train_df, test_df, target_class):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

        self.target_class = target_class

    def pre_processing(self):

        self.train_df['Loan_Status']  = (self.train_df.Loan_Status=='Y') * 1

        self.train_df['LoanAmount'].fillna(self.train_df['LoanAmount'].mean(), inplace=True)
        self.test_df['LoanAmount'].fillna(self.test_df['LoanAmount'].mean(), inplace=True)

        self.train_df['Loan_Amount_Term'].fillna(self.train_df['Loan_Amount_Term'].mean(), inplace=True)
        self.test_df['Loan_Amount_Term'].fillna(self.test_df['Loan_Amount_Term'].mean(), inplace=True)

        self.train_df['Credit_History'].fillna(-999, inplace=True)
        self.test_df['Credit_History'].fillna(-999, inplace=True)


        all_object_cols = self.get_all_object_cols()

        self.fill_nan(all_object_cols)
        self.encode_labels(all_object_cols)

    def get_mask(self):
        loantrain, loantest = train_test_split(xrange(self.train_df.shape[0]), train_size=0.7)
        loanmask = np.ones(self.train_df.shape[0], dtype='int')
        loanmask[loantrain] = 1
        loanmask[loantest] = 0
        loanmask = (loanmask==1)

        return loanmask

    def split_dataset(self):
        loanmask = self.get_mask()
        self.X_train = self.train_df[loanmask]
        self.y_train = self.train_df.Loan_Status[loanmask]

        self.X_val = self.train_df[~loanmask]
        self.y_val = self.train_df.Loan_Status[~loanmask]

    def get_cross_validation_scores(self, est, features, target_label):
        self.split_dataset()

        skf = StratifiedKFold(self.y_train, 5)
        X = self.X_train[features]
        y = self.y_train

        scores = cross_val_score(est, X, y, cv=skf)

        return scores

    def test(self, est, features):
        y_pred = est.predict(self.X_val[features])
        return accuracy_score(self.y_val, y_pred)

    def get_all_object_cols(self):
        return [col for col in self.train_df.columns.drop('Loan_Status') if self.train_df[col].dtype == 'O']

    def encode_labels(self, cols):
        for col in cols:

            lbl = LabelEncoder()
            feature = list(self.train_df[col].copy())
            feature.extend(self.test_df[col])
            lbl.fit(feature)

            self.train_df[col] = lbl.transform(self.train_df[col])
            self.test_df[col] = lbl.transform(self.test_df[col])

    def fill_nan(self, cols):
        for col in cols:
            self.train_df[col].fillna('-999', inplace=True)
            self.test_df[col].fillna('-999', inplace=True)

class LogRegression(Model):

    def __init__(self, train_df, test_df, target_class):
        super(LogRegression, self).__init__(train_df, test_df, target_class)


    def train_model(self, features, train_label):
        X = self.train_df[features]
        y = self.train_df[train_label]

        est = LogisticRegression(C=1.)
        est.fit(X, y)

        return est

    def predict(self, est, features):
        Xtest = self.test_df[features]

        log_reg_predictions = est.predict(Xtest)
        log_reg_predictions = map(inverse_mapping, log_reg_predictions)

        log_reg_predictions = np.array(log_reg_predictions)

        return log_reg_predictions

class RandomForestModel(Model):

    def __init__(self, train_df, test_df, target_class):
        super(RandomForestModel, self).__init__(train_df, test_df, target_class)


    def train_model(self, features, train_label):
        X = self.train_df[features]
        y = self.train_df[train_label]

        est = RandomForestClassifier(n_estimators=200, criterion='entropy', n_jobs=-1)
        est.fit(X, y)

        return est

    def predict(self, est, features):
        Xtest = self.test_df[features]

        log_reg_predictions = est.predict(Xtest)
        log_reg_predictions = map(inverse_mapping, log_reg_predictions)

        log_reg_predictions = np.array(log_reg_predictions)

        return log_reg_predictions

class GradientBoostingModel(Model):
    def __init__(self, train_df, test_df, target_class):
        super(GradientBoostingModel, self).__init__(train_df, test_df, target_class)


    def train_model(self, features, train_label):
        X = self.train_df[features]
        y = self.train_df[train_label]

        est = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, subsample=0.8, min_samples_leaf=10)
        est.fit(X, y)

        return est

    def predict(self, est, features):
        Xtest = self.test_df[features]

        log_reg_predictions = est.predict(Xtest)
        log_reg_predictions = map(inverse_mapping, log_reg_predictions)

        log_reg_predictions = np.array(log_reg_predictions)

        return log_reg_predictions


class BaseModel(Model):
    def __init__(self, train_df, test_df, target_class):
        super(BaseModel, self).__init__(train_df, test_df, target_class)

    def predict(self):
        most_prominent_class = np.argmax(self.train_df[self.target_class].value_counts())
        baseline_prediction = [most_prominent_class] * self.test_df.shape[0]
        baseline_prediction = map(inverse_mapping, baseline_prediction)

        baseline_prediction = np.array(baseline_prediction)

        return baseline_prediction

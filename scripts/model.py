from helper import inverse_mapping


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold


import numpy as np

class Model(object):

    def __init__(self, train_df, test_df, target_class):
        """
        Creates copy of both the training dataset
        and test dataset and stores column name of the
        target class.

        Input: train_df, test_df, target_class
        """

        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

        self.target_class = target_class

    def pre_processing(self):
        """
        Encodes the target variable from {Y, N} -> {1, 0}
        and replaces missing values with the mean value for all quantitative
        variables and with -999 for other categorical variables and encodes
        categorical variables.
        """

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

    def get_mask(self, dataset, train_size=0.7):
        """
        Returns the boolean mask which can be used to split the dataset
        """

        loantrain, loantest = train_test_split(xrange(dataset.shape[0]), train_size=train_size)
        loanmask = np.ones(dataset.shape[0], dtype='int')
        loanmask[loantrain] = 1
        loanmask[loantest] = 0
        loanmask = (loanmask==1)

        return loanmask

    def split_dataset(self):
        """
        Splits the training dataset into X_train and X_val
        """

        loanmask = self.get_mask(self.train_df)
        features = self.train_df.columns.drop('Loan_Status')

        self.X_train = self.train_df[features][loanmask]
        self.y_train = self.train_df.Loan_Status[loanmask]

        self.X_val = self.train_df[features][~loanmask]
        self.y_val = self.train_df.Loan_Status[~loanmask]

    def feature_selection(self):

        loanmask = self.get_mask(self.X_train)

        self.X_grid_search = self.X_train[loanmask]
        self.y_grid_search = self.y_train[loanmask]

        self.X_feature_selection = self.X_train[~loanmask]
        self.y_feature_selection = self.y_train[~loanmask]

        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        sel.fit(self.X_feature_selection)

        self.sel = sel


    def get_cross_validation_scores(self, est):
        sel = self.sel
        skf = StratifiedKFold(self.y_grid_search, 5)

        X = sel.transform(self.X_grid_search)
        y = self.y_grid_search

        scores = cross_val_score(est, X, y, cv=skf)

        return scores

    def fit_model(self, est):
        sel = self.sel

        X = sel.transform(self.X_grid_search)
        y = self.y_grid_search

        return est.fit(X, y)

    def test(self, est):
        sel = self.sel

        y_pred = est.predict(sel.transform(self.X_val))
        return accuracy_score(self.y_val, y_pred)

    def get_mispredicted_index(self, ytrue, ypred):
        return ytrue != ypred

    def analyze_mistakes(self, est):
        sel = self.sel

        y_pred = est.predict(sel.transform(self.X_val))
        return self.get_mispredicted_index(self.y_val, y_pred)

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


    def train_model(self, train_label):
        sel = self.sel
        features = self.train_df.columns.drop('Loan_Status')

        train_df = self.train_df[features]

        X = sel.transform(train_df)
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


    def train_model(self, train_label):
        sel = self.sel
        features = self.train_df.columns.drop('Loan_Status')

        train_df = self.train_df[features]

        X = sel.transform(train_df)
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


    def train_model(self, est, train_label):
        sel = self.sel
        features = self.train_df.columns.drop(train_label)

        train_df = self.train_df[features]

        X = sel.transform(train_df)
        y = self.train_df[train_label]

        est.fit(X, y)

        return est

    def predict(self, est):
        sel = self.sel
        Xtest = sel.transform(self.test_df)

        predictions = est.predict(Xtest)
        predictions = map(inverse_mapping, predictions)

        predictions = np.array(predictions)

        return predictions


class BaseModel(Model):
    def __init__(self, train_df, test_df, target_class):
        super(BaseModel, self).__init__(train_df, test_df, target_class)

    def predict(self):
        most_prominent_class = np.argmax(self.train_df[self.target_class].value_counts())
        baseline_prediction = [most_prominent_class] * self.test_df.shape[0]
        baseline_prediction = map(inverse_mapping, baseline_prediction)

        baseline_prediction = np.array(baseline_prediction)

        return baseline_prediction

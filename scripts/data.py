from sklearn.preprocessing import LabelEncoder

class Data(object):

    def __init__(self, train_df, test_df, target_label):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.target_label = target_label

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


    def get_train_X(self):
        features = self.train_df.columns.drop(self.target_label)

        return self.train_df[features]

    def get_train_Y(self):
        return self.train_df[self.target_label]

    def get_test_X(self):
        return self.test_df

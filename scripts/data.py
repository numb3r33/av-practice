class Data(object):

    def __init__(self, train_df, test_df, target_label):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.target_label = target_label

    def pre_processing(self):
        """
        Replaces missing values with the mean value for all quantitative
        variables and with -999 for other categorical variables and encodes
        categorical variables.
        """

        self.train_df['Loan_Status']  = (self.train_df.Loan_Status=='Y') * 1

        self.train_df['LoanAmount'].fillna(self.train_df['LoanAmount'].mean(), inplace=True)
        self.test_df['LoanAmount'].fillna(self.test_df['LoanAmount'].mean(), inplace=True)

        self.train_df['Loan_Amount_Term'].fillna(self.train_df['Loan_Amount_Term'].mean(), inplace=True)
        self.test_df['Loan_Amount_Term'].fillna(self.test_df['Loan_Amount_Term'].mean(), inplace=True)

        self.train_df['Gender'].fillna('-999', inplace=True)
        self.test_df['Gender'].fillna('-999', inplace=True)

        self.train_df['Married'].fillna('-999', inplace=True)
        self.test_df['Married'].fillna('-999', inplace=True)

        self.train_df['Dependents'].fillna('-999', inplace=True)
        self.test_df['Dependents'].fillna('-999', inplace=True)

        self.train_df['Self_Employed'].fillna('-999', inplace=True)
        self.test_df['Self_Employed'].fillna('-999', inplace=True)

        self.train_df['Credit_History'].fillna(-999, inplace=True)
        self.test_df['Credit_History'].fillna(-999, inplace=True)


    def get_train_X(self):
        features = self.train_df.columns.drop(self.target_label)

        return self.train_df[features]

    def get_train_Y(self):
        return self.train_df[self.target_label]

    def get_test_X(self):
        return self.test_df

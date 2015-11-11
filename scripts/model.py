from helper import inverse_mapping
import numpy as np

class Model(object):

    def __init__(self, train_df, test_df, target_class):
        self.train_df = train_df
        self.test_df = test_df

        self.target_class = target_class

class BaseModel(Model):
    def __init__(self, train_df, test_df, target_class):
        super(BaseModel, self).__init__(train_df, test_df, target_class)

    def predict(self):
        most_prominent_class = np.argmax(self.train_df[self.target_class].value_counts())
        baseline_prediction = [most_prominent_class] * self.test_df.shape[0]
        baseline_prediction = map(inverse_mapping, baseline_prediction)

        baseline_prediction = np.array(baseline_prediction)

        return baseline_prediction

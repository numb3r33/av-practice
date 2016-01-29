from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

import pandas as pd
import numpy as np

class FeatureTransformer(BaseEstimator):
	"""
	Generate features
	"""

	def __init__(self):
		pass

	def get_feature_names(self):
		feature_names = []

		feature_names.extend(['Number_Doses_Week',
			                  'Estimated_Insects_Count',
			                  'Number_Weeks_Quit',
			                  'Number_Weeks_Used',
			                  'Total_Dosage',
			             	  'Currently_Using_Pesticides',
			                  'Dosage_For_Insect_Count',
			                  'Dosage_Balance'
			                  ])

		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		numerical_features = self.get_numerical_features(X)
		
		features = []
		
		features.append(numerical_features)
		features = np.hstack(features)

		return features

	def get_numerical_features(self, X):
		Number_Doses_Week = np.log1p(X.Number_Doses_Week)
		Estimated_Insects_Count = np.log1p(X.Estimated_Insects_Count)
		Number_Weeks_Quit = np.log1p(X.Number_Weeks_Quit)
		Number_Weeks_Used = np.log1p(X.Number_Weeks_Used)
		Total_Dosage = Number_Doses_Week * Number_Weeks_Used

		Dosage_Balance = Number_Weeks_Used / Number_Weeks_Quit

		Currently_Using_Pesticides = (X.Pesticide_Use_Category==3) * 1.	
		Dosage_For_Insect_Count = Estimated_Insects_Count / Total_Dosage

		return np.array([Number_Doses_Week,
			             Estimated_Insects_Count,
			             Number_Weeks_Quit,
			             Number_Weeks_Used,
			             Total_Dosage,
			             Currently_Using_Pesticides,
			             Dosage_For_Insect_Count,
			             Dosage_Balance
			             ]).T
	

	def transform(self, X):
		numerical_features = self.get_numerical_features(X)
		features = []
		
		features.append(numerical_features)
		features = np.hstack(features)
		
		return features
	

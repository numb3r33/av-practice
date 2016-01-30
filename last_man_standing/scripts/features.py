from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans

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
			                  'Number_Weeks_Used',
			                  'Total_Dosage',
			                  'Zero_Weeks_Quit',
			                  'Zero_Weeks_Used',
			                  'Zero_Doses_Week',
			                  'Currently_Using_Pesticides',
			                  'Soil_Type',
			                  'Crop_Type',
			                  'Season_1',
			                  'Season_2'
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
		Number_Weeks_Used = np.log1p(X.Number_Weeks_Used)
		Total_Dosage = Number_Doses_Week * Number_Weeks_Used

		Zero_Weeks_Quit = (X.Number_Weeks_Quit==0) * 1.
		Zero_Weeks_Used = (X.Number_Weeks_Used==0.) * 1.
		Zero_Doses_Week = ((X.Number_Doses_Week==20) | (X.Number_Doses_Week==40)) * 1.

		Currently_Using_Pesticides = (X.Pesticide_Use_Category==3) * 1.
		
		Soil_Type = X.Soil_Type
		Crop_Type = X.Crop_Type

		Season_1 = (X.Season==1) * 1.
		Season_2 = (X.Season==2) * 1.


		return np.array([Number_Doses_Week,
			             Estimated_Insects_Count,
			             Number_Weeks_Used,
			             Total_Dosage,
			             Zero_Weeks_Quit,
			             Zero_Weeks_Used,
			             Zero_Doses_Week,
			             Currently_Using_Pesticides,
			             Soil_Type,
			             Crop_Type,
			             Season_1,
			             Season_2
			             ]).T
	

	def transform(self, X):
		numerical_features = self.get_numerical_features(X)
		features = []

		features.append(numerical_features)
		features = np.hstack(features)
		
		return features
	

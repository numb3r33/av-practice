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
			                  'Zero_Weeks_Used',
			                  'Zero_Doses_Week',
			                  'Lower_Insect_Count',
			                  'Soil_Type',
			                  'Crop_Type',
			                  ])

		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		numerical_features = self.get_numerical_features(X)
		# pesticide_use_features = pd.get_dummies(X.Pesticide_Use_Category, prefix='Pest')
		# season_features = pd.get_dummies(X.Season, prefix='Season')

		features = []
		
		features.append(numerical_features)
		# features.append(pesticide_use_features)
		# features.append(season_features)


		features = np.hstack(features)

		return features

	def get_numerical_features(self, X):

		Number_Doses_Week = X.Number_Doses_Week
		Estimated_Insects_Count = X.Estimated_Insects_Count
		Number_Weeks_Used = X.Number_Weeks_Used
		Number_Weeks_Quit = X.Number_Weeks_Quit

		Currently_Using_Pesticides = (X.Pesticide_Use_Category==3) * 1.
		Soil_Type = X.Soil_Type
		Crop_Type = X.Crop_Type

		# group_by_num_doses_week = X.groupby('Number_Doses_Week')['Estimated_Insects_Count'].mean()
		# estimated_insects_count_per_pest = X.groupby('Pesticide_Use_Category')['Estimated_Insects_Count'].mean()
		# estimated_insect_group_per_crop_type = X.groupby('Crop_Type')['Estimated_Insects_Count'].mean()
		# estimated_insect_group_per_season = X.groupby('Season')['Estimated_Insects_Count'].mean()

		def avg_insect_count_per_dose(row):
			num_doses = row.Number_Doses_Week
			return group_by_num_doses_week.ix[num_doses] - row.Estimated_Insects_Count

		def avg_insect_count_per_pesticide_strategy(row):
			pest_used = row.Pesticide_Use_Category
			return estimated_insects_count_per_pest.ix[pest_used] - row.Estimated_Insects_Count

		def avg_insect_count_per_crop_type(row):
			crop_type = row.Crop_Type
			return int(estimated_insect_group_per_crop_type.ix[crop_type] < row.Estimated_Insects_Count)

		def avg_insect_count_per_season(row):
			season = row.Season
			return int(estimated_insect_group_per_season.ix[season] < row.Estimated_Insects_Count)


		insect_value_counts = X.Estimated_Insects_Count.value_counts()
		num_doses_week_value_counts = X.Number_Doses_Week.value_counts()

		Insect_Count = X.Estimated_Insects_Count.map(lambda x: insect_value_counts.ix[x])
		Pesticide_Not_Used_Despite_High_Insect_Count = ((X.Estimated_Insects_Count > X.Estimated_Insects_Count.mean()) & (X.Pesticide_Use_Category==1)) * 1.
		# Avg_insect_count_per_dose = X.apply(avg_insect_count_per_dose, axis=1)
		# Avg_insect_count_per_pesticide_strategy = X.apply(avg_insect_count_per_pesticide_strategy, axis=1)
		# Avg_insect_count_per_crop_type = X.apply(avg_insect_count_per_crop_type, axis=1)
		# Avg_insect_count_per_season = X.apply(avg_insect_count_per_season, axis=1)

		# Total_Dosage = Number_Doses_Week * Number_Weeks_Used

		# Zero_Weeks_Used = (X.Number_Weeks_Used==0) * 1.
		# Zero_Doses_Week = (X.Number_Doses_Week==0) * 1.
		# Soil_Type = X.Soil_Type
		# Crop_Type = X.Crop_Type

		return np.array([
						 Number_Doses_Week,
			             Estimated_Insects_Count,
			             Number_Weeks_Used,
			             Number_Weeks_Quit,
			             Currently_Using_Pesticides,
			             Soil_Type,
			             Crop_Type,
			             Insect_Count,
			             Pesticide_Not_Used_Despite_High_Insect_Count,
			             # Avg_insect_count_per_dose,
			             # Avg_insect_count_per_pesticide_strategy,
			             # Avg_insect_count_per_crop_type,
			             # Avg_insect_count_per_season,
			             # Total_Dosage,
			             # Zero_Weeks_Used,
			             # Zero_Doses_Week,
			             ]).T
	

	def transform(self, X):
		numerical_features = self.get_numerical_features(X)
		# pesticide_use_features = pd.get_dummies(X.Pesticide_Use_Category, prefix='Pest')
		# season_features = pd.get_dummies(X.Season, prefix='Season')

		features = []

		features.append(numerical_features)
		# features.append(pesticide_use_features)
		# features.append(season_features)

		features = np.hstack(features)
		
		return features
	

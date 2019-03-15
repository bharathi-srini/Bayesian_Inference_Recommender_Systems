import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

import create_data, feature_engineering, item2vec_embedding

def read_data:
	"""
	Function to invoke create_data script to read in 
	Instacart csv files, merge and write dataframe to 
	folder
	"""
	create_data.prepare_data()


def create_features(df):
	"""
	Function to create user, product and interaction features
	Arg : data frame
	Writes csv to folder
	"""
	feature_engineering.create_all(df)


def sample_data(fraction):
	"""
	Reads in merged data from drive
	Samples a smaller fraction of data for use
	Returns smaller dataframe
	"""
	folder = '/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'
	df_big = pd.read_csv(folder + 'merged_data.csv')
	return df_big.sample(frac = fraction, random_state = 100)


def main():
	#read_data()

	# Sample smaller data
	df = sample_data(fraction = 0.01)

	#Add features to data
	df1 = create_features(df)




if __name__ == '__main__':
	main()
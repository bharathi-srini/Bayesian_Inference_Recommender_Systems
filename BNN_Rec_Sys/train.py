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


def sample_data(fraction):
	"""
	Reads in merged data from drive
	Samples a smaller fraction of data for use
	Returns smaller dataframe
	"""
	folder = '/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'
	df_big = pd.read_csv(folder + 'merged_data.csv')
	return df_big.sample(frac = fraction, random_state = 100)


def add_embeddings(df):
	"""
	Invoke item2vec model from gensim on products
	Train with basket as context
	Add embeddings to dataframe and return result
	"""
	item2vec_model = item2vec_embeddings.generate_prod_embeddings(df)
	df['prod_embedding'] = item2vec_model.wv(df['product_name'])
	return df


def main():
	#read_data()

	# Sample smaller data
	df = sample_data(fraction = 0.01)

	#Add features to data
	df1 = feature_engineering.create_all(df)

	# Adding product embeddings to data
	df2 = add_embeddings(df1)
	print(head(df2))





if __name__ == '__main__':
	main()
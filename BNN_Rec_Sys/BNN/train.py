import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

import Data.create_data as initialise
import Data.feature_engineering as features
import Embedding.predictNN_embedding as embed
from utils.utils import val2idx

folder = '/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'

def read_data():
	"""
	Function to invoke create_data script to read in 
	Instacart csv files, merge and write dataframe to 
	folder
	"""
	initialise.prepare_data(folder)

def data_nusers(df, n):
	"""
	Sample data by choosing all orders of n users
	"""
	unique_users = df.user_id.unique()
	i = 0
	df_nusers = pd.DataFrame()  
	for user in unique_users:
		df_nusers = df_nusers.append(df[df.user_id == user])
		i +=1
		if (i == n):
			break
	return pd.DataFrame(df_nusers)

def sample_data(fraction):
	"""
	Reads in merged data from drive
	Samples a smaller fraction of data for use
	Returns smaller dataframe
	"""
	#folder = '/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'
	df_big = pd.read_csv(folder + 'merged_data.csv')
	return df_big.sample(frac = fraction, random_state = 100)

def add_embeddings(df):
	"""
	Invoke item2vec model from gensim on products
	Train with basket as context
	Add embeddings to dataframe and return result
	"""
	item2vec_model = item2vec_embedding.generate_prod_embeddings(df)
	df['prod_embedding'] = item2vec_model.wv(df['product_name'])
	return df

def train_embeddings_model(df):
	EMBEDDING_COLUMNS = ["user_id", "product_id"]
	df_use = utils.val2idx(df, EMBEDDING_COLUMNS)
	
	product_in , user_in, basket_in, predicted_product = embed.create_sequence(df_use)

	# Fitting model to data
	embed.create_embedding_network(N_products, N_shoppers, product_in , user_in, basket_in, predicted_product )


def main():
	#read_data()

	# Sample smaller data
	'''
	df = sample_data(fraction = 1)
	print('Size of sample :' ,df.shape)
	print('Unique users: ' ,df.user_id.nunique())
	print('Unique products: ' ,df.product_id.nunique())
	print('Unique orders: ' ,df.order_id.nunique())


	#train_embeddings_model(df_10users)
	'''

	df = sample_data(fraction=1)
	df_use = data_nusers(df, 300)
	print('Size of sample:', df_use.shape)
	print('users :', df_use.user_id.nunique())
	print('products: ', df_use.product_id.nunique() )

	'''
	out_sample_users = pd.read_csv(folder+'out_sample_users.csv')
	out_sample_prd = pd.read_csv(folder + 'out_sample_prd.csv')
	out_sample_both = pd.read_csv(folder+'out_sample_both.csv')
	#Add features to data
	df1 = features.create_all(out_sample_users)
	print('Feature engineering done - users')
	df1.to_csv(folder + 'engineered_data_users_out.csv', index = False)

	df2 = features.create_all(out_sample_prd)
	print('Feature engineering done - prd')
	df2.to_csv(folder + 'engineered_data_out_prd.csv', index = False)


	df3 = features.create_all(out_sample_both.sample(frac=0.01))
	print('Feature engineering done - both')
	df3.to_csv(folder + 'engineered_data_out_both.csv', index = False)
	'''

	df1 = features.create_all(df_use)
	print('Feature engineering done')
	df1.to_csv(folder + 'engineered_data_300.csv', index = False)

	#plt.figure()
	#plt.hist(df1.reordered)
	#plt.savefig('reordered_dist.pdf')

	# Adding product embeddings to data
	#df2 = item2vec_embedding.generate_prod_embeddings(df1)
	#df1.to_csv('/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'+'data_final.csv', index=False)
	#print('data written to file')





if __name__ == '__main__':
	main()
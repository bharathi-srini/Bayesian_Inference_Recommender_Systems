import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

import create_data, feature_engineering, item2vec_embedding

def read_data():
	"""
	Function to invoke create_data script to read in 
	Instacart csv files, merge and write dataframe to 
	folder
	"""
	create_data.prepare_data()

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
	folder = '/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'
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


def main():
	#read_data()

	# Sample smaller data
	df = sample_data(fraction = 0.001)
	print('Size of sample :' ,df.shape)

	#Add features to data
	df1 = feature_engineering.create_all(df)
	print('Feature engineering done')

	plt.figure()
	plt.hist(df1.reordered)
	plt.savefig('reordered_dist.pdf')



	# Adding product embeddings to data
	#df2 = item2vec_embedding.generate_prod_embeddings(df1)
	df1.to_csv('/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'+'data_final.csv', index=False)
	print('data written to file')



if __name__ == '__main__':
	main()
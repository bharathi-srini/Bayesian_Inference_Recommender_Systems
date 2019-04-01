import pandas as pd
import sys
sys.path.append('../')

from utils.utils import val2idx
import predictNN_embedding as embed


folder = 'C:\\Users\\Pascal\\Documents\\GitHub\\instacart-market-basket-analysis\\'

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
	df_use = val2idx(df, EMBEDDING_COLUMNS)
	
	transformed_dat, basket, N_products, N_shoppers = embed.transform_data_for_embedding(df_use)
	product_in , user_in, basket_in, predicted_product = embed.create_input_for_embed_network(df_deep, df1, basket, N_products)

	# Fitting model to data
	embed.create_embedding_network(N_products, N_shoppers, product_in, user_in, basket_in, predicted_product )

def data_nusers(df, n):
	"""
	Sample data by choosing all orders of n users
	"""
	unique_users = df.user_id.unique()
	i = 0
	df_nusers = pd.DataFrame()
	for user in unique_users:
		df_nusers.append(df[df.user_id == user])
		i +=1
		if (i == n):
			break
	return pd.DataFrame(df_nusers)

def main():
	df = pd.read_csv(folder + 'data1000.csv')
	#df_use = data_nusers(df_big, 1000)
	#print('Size of data with 1000 users is: ', df_use.shape)
	#df_use.to_csv(folder+'data1000.csv', index=False)

	train_embeddings_model(df)


if __name__ == '__main__':
	main()
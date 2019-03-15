import pandas as pd
import gc


def prod_features(df):
	"""
	Product Features
	"""
	sub1 = df.sort_values(['product_id'],ascending=True)

	# 'product_id_orders' indicates the popularity of a product
	sub2 = sub1.join(sub1.groupby('product_id')['product_id'].size(), on='product_id', rsuffix='_orders')

	# 'reordered_total' indicates how many times a product was reordered
	sub3 = sub2.join(sub2.groupby('product_id')['reordered'].sum(), on='product_id', rsuffix='_total')

	del sub1,sub2
	gc.collect()

	return sub3[['user_id','product_id', 'product_id_orders','reordered_total']]


def user_features(df):
	"""
	User features from df dataframe
	returns users features by id
	"""
	# Total orders by user
	users = df.groupby(['user_id'])['order_number'].max().to_frame('user_orders')

	# Not ideal - but an indidcation of customer's history with the business
	users['user_period'] = df.groupby(['user_id'])['days_since_prior_order'].sum()

	# Average time taken by a user before returning for a purchase
	users['avg_time_to_order'] = df.groupby(['user_id'])['days_since_prior_order'].mean()

	# Total unique products ordered by a customer
	users['user_distinct_products'] = df.groupby(['user_id'])['product_id'].nunique()
	 
	# Total products ordered by a customer
	users['total_products'] = df.groupby(['user_id'])['product_id'].size()

	# the average basket size of the user
	users['user_average_basket'] = users['total_products'] / users['user_orders']

	return users


def interaction_features(df):
	"""
	User-Product Features
	Input: dataframe
	Outut: features columns
	"""

	# Product Reorder Frequency indicates a user's preference for a particular product
	df['prd_reorder_freq'] = df[df['reordered']==1].groupby(['user_id', 'product_id']).cumcount()+1

	# Average position of the product in a user's cart
	df1 = df.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_cart_position')

	return df1


def create_all(df):
	"""
	Merge all features to create final dataframe
	"""

	df1 = interaction_features(df).reset_index()
	df2 = pd.merge(df, df1, how='left', on=['user_id', 'product_id'])

	users = user_features(df).reset_index()
	df3 = pd.merge(df2, users, how='left', on='user_id')

	prd = prod_features(df).reset_index()
	df_final = pd.merge(df3, prd, how ='left', on='product_id')
	
	df_final.drop(['Unnamed: 0', 'user_id_y'], axis=1, inplace=True)

	del df,df1,df2,df3,users,prd
	gc.collect()

	return df_final







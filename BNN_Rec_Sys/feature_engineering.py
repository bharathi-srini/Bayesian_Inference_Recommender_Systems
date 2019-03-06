import pandas as pd
import gc

# Change folder to location of output of create_data.py
folder = '/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'
df = pd.read_csv(folder + 'merged_data.csv')

################ Product Features ############################################################

sub1 = df.sort_values(['product_id'],ascending=True)

# 'product_id_orders' indicates the popularity of a product
sub2 = sub1.join(sub1.groupby('product_id')['product_id'].size(), on='product_id', rsuffix='_orders')

# 'reordered_total' indicates how many times a product was reordered
sub3 = sub2.join(sub2.groupby('product_id')['reordered'].sum(), on='product_id', rsuffix='_total')

prd = sub3[['user_id','product_id', 'product_id_orders','reordered_total']]

del sub1,sub2,sub3
gc.collect()

print('Product related features created!')

############# User Features ##################################################################

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

print('User related features created!')

########### User-Product Features ###########################################################

# Product Reorder Frequency indicates a user's preference for a particular product
df['prd_reorder_freq'] = df[df['reordered']==1].groupby(['user_id', 'product_id']).cumcount()+1

# Average position of the product in a user's cart
df1 = df.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('average_cart_position')

print('User-Product Features created!')
########### Merge all features to create final dataframe ####################################

df1 = df1.reset_index()
df2 = pd.merge(df, df1, how='left', on=['user_id', 'product_id'])

users = users.reset_index()
df3 = pd.merge(df2, users, how='left', on='user_id')

prd = prd.reset_index()
df_final = pd.merge(df3, prd, how ='left', on='product_id')

df_final.drop(['Unnamed: 0', 'user_id_x','index_x','level_0','index_y','user_id_y'], axis=1, inplace=True)

# Export dataset for future use
print('Final dataset writing to csv')

outfile = open(folder+'final_data.csv', 'wb')
df_final.to_csv('final_data.csv',index = False, header = True, sep = ',', encoding = 'utf-8')
outfile.close()

del df,df1,df2,df3,users,prd
gc.collect()







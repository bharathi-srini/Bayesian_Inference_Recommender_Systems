import pandas as pd

# Change folder to location of Instacart Data!
folder = '/Users/BharathiSrinivasan/Documents/Instacart_Data/'

# Reading in csv
prior = pd.read_csv(folder + "order_products__prior.csv")
train = pd.read_csv(folder + "order_products__train.csv")
orders = pd.read_csv(folder + "orders.csv")
products = pd.read_csv(folder + "products.csv")
dept = pd.read_csv(folder + "departments.csv")

# We will create a merged dataset from which we can split data into test
# and train datasets of different sizes
df = pd.concat([prior, train])

df1 = pd.merge(df, orders, how='left', left_on='order_id', right_on='order_id')
df2 = pd.merge(df1, products, how='left', left_on='product_id', right_on='product_id')
df_merged = pd.merge(df2, dept, how='left', left_on='department_id', right_on='department_id')

# Eval set column indicates which dataset the row belonged to - we don't need this!
df_merged.drop(['eval_set'], axis=1, inplace=True)

# 2 million rows with missing values in days since prior order - Impute with mean
df_merged['days_since_prior_order'] = df_merged['days_since_prior_order'].fillna(value=df_merged['days_since_prior_order'].mean())

# Export dataset
df_merged.to_csv(folder + 'merged_data.csv')

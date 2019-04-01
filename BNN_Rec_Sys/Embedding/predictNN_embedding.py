from keras.models import Model
from keras.layers.core import Dense, Reshape, Lambda
from keras.layers import Input, Embedding, merge
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import preprocessing
from keras.regularizers import l2
import random
from keras.layers.advanced_activations import LeakyReLU


def first_prod(order):
    for _,row in order.iterrows():
        if row['add_to_cart_order']==1:
            return row['product_id']

def next_prod(order):
    for _,row in order.iterrows():
        if row['add_to_cart_order']==2:
            return row['product_id']


def transform_data_for_embedding(df):
    first = df.groupby(['order_id']).apply(first_prod)
    next_product = df.groupby(['order_id']).apply(lambda x:next_prod(x))
    basket =df.groupby(['order_id', 'product_id']).size().unstack(fill_value=0)
    transform_df = pd.DataFrame(first, columns = ['first_prod'])
    transform_df['next_product']= next_product.values
    transform_df.reset_index(inplace=True)

    # Number of product IDs available
    N_products = df['product_id'].nunique()
    N_shoppers = df['user_id'].nunique()

    return transform_df, basket, N_products, N_shoppers

def create_input_for_embed_network(df, transform_df, basket, N_products):

    # Creating df with order_id, user_id, first prod, next prod, basket 
    x = df.drop_duplicates(subset=['order_id','user_id'])
    train_df = pd.merge(transform_df, x[['order_id','user_id']], how='left', on='order_id' )
    train_df.dropna(inplace=True)
    
    basket.reset_index(inplace=True)
    basket_df = pd.merge(train_df[['order_id']], basket, how='left', on ='order_id')
    basket_df.drop(['order_id'], axis=1, inplace=True)

    train_df['next_product'] = train_df['next_product'].astype('category', categories = df.product_id.unique())
    y_df = pd.get_dummies(train_df, columns = ['next_product'])
    y_df.drop(['user_id','order_id','first_prod'], axis=1, inplace=True)

    return train_df['first_prod'], train_df['user_id'], basket_df, y_df

def create_embedding_network(N_products, N_shoppers, prior_in, shopper_in, candidates_in, predicted ):

    # Integer IDs representing 1-hot encodings
    prior_in = Input(shape=(1,))
    shopper_in = Input(shape=(1,))

    # Dense N-hot encoding for candidate products
    candidates_in = Input(shape=(N_products,))

    # Embeddings
    prior = Embedding(N_products+1, 10)(prior_in)
    shopper = Embedding(N_shoppers+1, 10)(shopper_in)

    # Reshape and merge all embeddings together
    reshape = Reshape(target_shape=(10,))
    combined = merge([reshape(prior), reshape(shopper)],
                 mode='concat')

    # Hidden layers
    #hidden_1 = Dense(1024, activation='relu',W_regularizer=l2(0.02))(combined)
    #hidden_2 = Dense(512, activation='relu',W_regularizer=l2(0.02))(hidden_1)
    hidden_3 = Dense(100, activation='relu')(combined)
    hidden_4 = Dense(1, activation='relu')(hidden_3)

    # Final 'fan-out' into the space of future products
    final = Dense(N_products, activation='relu')(hidden_4)

    # Ensure we do not overflow when we exponentiate
    final = Lambda(lambda x: x - K.max(x))(final)

    # Masked soft-max using Lambda and merge-multiplication
    exponentiate = Lambda(lambda x: K.exp(x))(final)
    masked = merge([exponentiate, candidates_in], mode='mul')
    predicted = Lambda(lambda x: x / K.sum(x))(masked)

    # Compile with categorical crossentropy and adam
    mdl = Model(input=[prior_in, shopper_in, candidates_in],
            output=predicted)
    mdl.compile(loss='categorical_crossentropy', 
            optimizer='adam',
            metrics=['accuracy'])

    mdl.fit([prior_in, shopper_in, candidates_in], predicted,  batch_size=128, epochs=3, verbose=1)
    
    # serialize model to JSON
    model_json = mdl.to_json()
    with open("NN_embed_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    mdl.save_weights("NN_embed_model.h5")
    print("Saved model to disk")





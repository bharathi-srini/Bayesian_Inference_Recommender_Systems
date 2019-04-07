import keras
from keras.models import Model
from keras.layers.core import Dense, Reshape, Lambda
from keras.layers import Input, Embedding, merge, Multiply, Concatenate
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import preprocessing
from keras.regularizers import l2
import random


import sys
sys.path.append('../')

#from BNN.train import sample_data
import talos


def create_sequence(df_deep):
    """
    For each order, first prod is product added first
    next_prod is the subsequent product added to basket
    Iterates sequentially for all products in the basket
    """
    # Creating an order size columns for creating purchase sequences later
    order_size = pd.DataFrame(df_deep.groupby(['order_id']).size())
    order_size = order_size.reset_index()
    order_size.columns = {'order_id','size'}

    df_deep = df_deep.merge(order_size, how='left', on='order_id')

    # For every order, we create a row with product ordered and the next product added to basket
    orders_list = df_deep.order_id.unique()
    first_prod = []
    next_prod = []
    order_ids = []

    for order in orders_list:
        temp = df_deep[df_deep.order_id == order]
        for i in range(int(temp.size) - 1):
            order_ids.append((temp[temp.add_to_cart_order == i].order_id).values)
            first_prod.append((temp[temp.add_to_cart_order == i].product_id).values)
            next_prod.append((temp[temp.add_to_cart_order == i+1].product_id).values)

    # Creating dataframe with these sequences
    sequence_df = pd.DataFrame(columns={'order_id','first_prod','next_prod'} )
    sequence_df.order_id = order_ids
    sequence_df.first_prod = first_prod
    sequence_df.next_prod = next_prod

    for col in sequence_df.columns:
        sequence_df[col] = sequence_df[col].apply(pd.Series)
    
    #first product of basket creates empty row
    sequence_df = sequence_df.dropna()

    merged1 = pd.merge(sequence_df, df_deep[['order_id', 'user_id']], how='left', on='order_id')
    merged1 = merged1.drop_duplicates()

    basket =df_deep.groupby(['order_id', 'product_id']).size().unstack(fill_value=0)
    basket = basket.reset_index()

    #Making dataframe sizes equal with a merge
    merged = pd.merge(sequence_df, basket, how='left', on='order_id')

    product_in = merged['first_prod']
    user_in = merged1['user_id']

    basket = merged.drop(['Unnamed: 0', 'order_id','first_prod', 'next_prod'], axis=1)

    sequence_df['next_prod'] = sequence_df['next_prod'].astype('category', categories = df_deep.product_id.unique())
    predicted_product = pd.get_dummies(sequence_df, columns = ['next_prod'])
    predicted_product.drop(['Unnamed: 0','order_id','first_prod'], axis=1, inplace=True)

    return product_in, user_in, basket, predicted_product



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
    combined = keras.layers.concatenate([reshape(prior), reshape(shopper)])

    # Hidden layers
    hidden_1 = Dense(512, activation='relu')(combined)
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    hidden_3 = Dense(128, activation='relu')(hidden_2)
    hidden_4 = Dense(50, activation='linear')(hidden_3)

    # Final 'fan-out' into the space of future products
    final = Dense(N_products, activation='linear')(hidden_4)

    # Ensure we do not overflow when we exponentiate
    final = Lambda(lambda x: x - K.max(x))(final)

    # Masked soft-max using Lambda and merge-multiplication
    exponentiate = Lambda(lambda x: K.exp(x))(final)
    masked = keras.layers.multiply([exponentiate, candidates_in])
    predicted = Lambda(lambda x: x / K.sum(x))(masked)

    # Compile with categorical crossentropy and adam
    mdl = Model(input=[prior_in , shopper_in, candidates_in],output=predicted)
    mdl.compile(loss='categorical_crossentropy', \
            optimizer=optimizers.Adam(),metrics=['accuracy'])

    history_callback = mdl.fit([prior_in, shopper_in, candidates_in], predicted,  batch_size=128, epochs=3, verbose=1)
    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["accuracy"]
    loss_history_array = np.array(loss_history)
    acc_history_array = np.array(acc_history)
    np.savetxt(folder+"loss_history.txt", loss_history_array, delimiter=",")
    np.savetxt(folder+"acc_history.txt", acc_history_array, delimiter=",")

    # serialize model to JSON
    model_json = mdl.to_json()
    with open(folder+"NN_embed_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    mdl.save_weights(folder+"NN_embed_model.h5")
    print("Saved model to disk")








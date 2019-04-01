import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
%matplotlib inline

import keras
from keras import backend as K
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input,Activation, concatenate, Embedding, Reshape
from keras.layers import Flatten, merge, Lambda, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from keras.regularizers import l1,l2
import tensorflow as tf
import gc

from utils import val2idx

def load_embeddings_model():
	"""
	Load the pre-trained émbeddings model and extract for users and products
	"""
	# load embeddings model
	json_file = open('NN_embed_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("NN_embed_model.h5")
	print("Loaded model from disk")

	# Exctracted embeddings from pr-trained model
	embeddings_prior = loaded_model.layers[2].get_weights()[0]
	embeddings_user = loaded_model.layers[3].get_weights()[0]

	return embeddings_prior, embeddings_user

def prepare_data_for_NN(df_use):
	EMBEDDING_COLUMNS = ["user_id", "product_id"]
	CATEGORICAL_COLUMNS = ["order_dow", "order_hour_of_day","aisle_id","department_id"]
	CONTINUOUS_COLUMNS = ["days_since_prior_order","order_number","add_to_cart_order", \
						"user_orders", "user_period", "product_id_orders"]
	df_deep, values = val2idx(df_use, EMBEDDING_COLUMNS)

	df_deep.drop(['product_name','department','prd_reorder_freq', "index"], axis=1, inplace=True)

	#One-hot encoding categorical columns
	df_deep = pd.get_dummies(df_deep, columns=[x for x in CATEGORICAL_COLUMNS])

	#Normalising the feature columns
	df_deep[CONTINUOUS_COLUMNS] = MinMaxScaler().fit_transform(df_deep[CONTINUOUS_COLUMNS].values)

	return df_deep

def test_train_split_data(df):
	y = df.reordered.values
	df.drop(['reordered'], axis=1, inplace = True)

	X_train, X_test, y_train, y_test = train_test_split(df_small, y, \
														test_size=0.20, random_state=42, stratify=y)

	products_in = X_train['product_id']
	X_train.drop(['product_id'], axis = 1, inplace=True)

	products_test = X_test['product_id']
	X_test.drop(['product_id'], axis = 1, inplace=True)

	users_in = X_train['user_id']
	X_train.drop(['user_id'], axis=1, inplace= True)

	users_test = X_test['user_id']
	X_test.drop(['user_id'], axis=1, inplace= True)

	return products_in, users_in, X_train, y_train, products_test, users_test, X_test, y_test

def create_MCdropout_model():
	# Integer IDs representing 1-hot encodings
	prior_in = Input(shape=(1,))
	shopper_in = Input(shape=(1,))

	# Embeddings
	prior = Embedding(input_dim=embeddings_prior.shape[0], input_length=1, weights=[embeddings_prior],\
						output_dim=embeddings_prior.shape[1], trainable=False)(prior_in)
	shopper = Embedding(input_dim=embeddings_user.shape[0], input_length=1, weights=[embeddings_user],\
						output_dim=embeddings_prior.shape[1], trainable=False)(shopper_in)

	# Numeric and categorical inputs
	input_tensor = Input(shape=X_train.shape[1:])

	reshape = Reshape(target_shape=(10,))
	combined_input = keras.layers.concatenate([reshape(prior), reshape(shopper), input_tensor])

	x = BatchNormalization()(combined_input)

	x = Dropout(0.5)(x, training=True)
	x = Dense(5, activation='relu')(x)
	x = BatchNormalization()(x)

	x = Dropout(0.5)(x, training=True)
	x = Dense(2, activation='relu')(x)
	x = BatchNormalization()(x)

	logits = Dense(1,activation='relu')(x)

	softmax_output = Activation('softmax', name='softmax_output')(logits)

	model = Model(inputs=[prior_in, shopper_in, input_tensor], outputs=softmax_output)

	return model


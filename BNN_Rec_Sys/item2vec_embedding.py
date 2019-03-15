import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def get_batch(vocab, model, n_batches=4):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    f = plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
#     plt.savefig(filename)
    plt.show('hold')
    f.savefig('embeddings.pdf')


def find_similar(model, word):
	return model.wv.most_similar(positive = word, topn = 5)


def generate_prod_embeddings(df_small):

	df_small['product_id'] = df_small['product_id'].astype(str)

	#Baskets will be used as sentences
	baskets = df_small.groupby('order_id').apply(lambda order: order['product_id'].tolist())

	longest = np.max(baskets.apply(len))
	baskets = baskets.values
	print('sentences created')

	#Word2vec on baskets
	model = gensim.models.Word2Vec(baskets, size=100, window=10, min_count=2, workers=4)
	model.train(baskets, total_examples=len(baskets), epochs=10)
	vocab = list(model.wv.vocab.keys())

	# Saving model for training later with new words
	model.save("word2vec.model")

	my_dict = dict({})
	for idx, key in enumerate(model.wv.vocab):
		my_dict[key] = model.wv[key]

	df_small['prod_embedding'] = df_small['product_id'].map(my_dict)
	
	"""
	PCA and plotting

	pca = PCA(n_components=2)
	pca.fit(model.wv.syn0)
	
	products = pd.read_csv('/Users/BharathiSrinivasan/Documents/Instacart_Data/products.csv')

	embeds = []
	labels = []
	for item in get_batch(vocab, model, n_batches=4):
	    embeds.append(model[item])
	    labels.append(products.loc[int(item)]['product_name'])
	embeds = np.array(embeds)
	embeds = pca.fit_transform(embeds)
	plot_with_labels(embeds, labels)
	"""
	return df_small


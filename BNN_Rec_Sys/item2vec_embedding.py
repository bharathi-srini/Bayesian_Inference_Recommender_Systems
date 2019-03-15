import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

folder = '/Users/BharathiSrinivasan/Documents/GitHub/Thesis/'
df = pd.read_csv(folder + 'merged_data.csv')
df_small = df.sample(frac = 0.01)
print('df read')

df_small['product_id'] = df_small['product_id'].astype(str)

#Baskets will be used as sentences
baskets = df_small.groupby('order_id').apply(lambda order: order['product_id'].tolist())

longest = np.max(baskets.apply(len))
baskets = baskets.values
print('sentences created')

#Word2vec on baskets
model = gensim.models.Word2Vec(baskets, size=100, window=longest, min_count=2, workers=4)
vocab = list(model.wv.vocab.keys())

print('Embedding created for products!')

#PCA and plotting
pca = PCA(n_components=2)
pca.fit(model.wv.syn0)

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
    f.savefig('embeddings4.pdf')

products = pd.read_csv("/Users/BharathiSrinivasan/Documents/Instacart_Data/products.csv")

embeds = []
labels = []
for item in get_batch(vocab, model, n_batches=4):
    embeds.append(model[item])
    labels.append(products.loc[int(item)]['product_name'])
embeds = np.array(embeds)
embeds = pca.fit_transform(embeds)
plot_with_labels(embeds, labels)

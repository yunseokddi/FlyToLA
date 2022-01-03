import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition
from scipy import linalg

np.set_printoptions(suppress=True)

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
remove = ('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

# print(newsgroups_train.filenames.shape, newsgroups_train.target.shape)
# # (2034,) (2034,)

# print("\n".join(newsgroups_train.data[:3]))
# # Hi,
# #
# # I've noticed that if you only save a model (with all your mapping planes
# # positioned carefully) to a .3DS file that when you reload it after restarting
# # 3DS, they are given a default position and orientation.  But if you save
# # to a .PRJ file their positions/orientation are preserved.  Does anyone
# # know why this information is not stored in the .3DS file?  Nothing is
# # explicitly said in the manual about saving texture rules in the .PRJ file.
# # I'd like to be able to read the texture rule information, does anyone have
# # the format for the .PRJ file?

# print(np.array(newsgroups_train.target_names)[newsgroups_train.target[:3]])
# # ['comp.graphics' 'talk.religion.misc' 'sci.space']

num_topics, num_top_words = 6, 8

vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(newsgroups_train.data).todense()
# print(vectors.shape)
# # (2034, 26576)

# print(len(newsgroups_train.data), vectors.shape)
# # 2034 (2034, 26576)

vocab = np.array(vectorizer.get_feature_names())
# print(vocab.shape)
# # (26576,)

# print(vocab[7000:7020])
# # ['cosmonauts' 'cosmos' 'cosponsored' 'cost' 'costa' 'costar' 'costing'
# #  'costly' 'costruction' 'costs' 'cosy' 'cote' 'couched' 'couldn' 'council'
# #  'councils' 'counsel' 'counselees' 'counselor' 'count']

# SVD
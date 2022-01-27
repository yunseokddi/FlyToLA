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

num_topics, num_top_words = 6, 8

vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(newsgroups_train.data)

vocab = np.array(vectorizer.get_feature_names())

# m, n = vectors.shape
d = 5  # num topics

clf = decomposition.NMF(n_components=d, random_state=1)


# W1 = clf.fit_transform(vectors)
# H1 = clf.components_

def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words - 1: -1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]


# print(show_topics(H1))
# plt.plot(clf.components_[0])
# plt.show()
# print(clf.reconstruction_err_)

# Goal: Minimized the Frobenius norm of V - WH

vectorizer_tfids = TfidfVectorizer(stop_words='english')
vectors_tfidf = vectorizer_tfids.fit_transform(newsgroups_train.data)

# W1 = clf.fit_transform(vectors_tfidf)
# H1 = clf.components_

lam = 1e3
lr = 1e-2
m, n = vectors_tfidf.shape

W1 = clf.fit_transform(vectors)
H1 = clf.components_
print(show_topics(H1))

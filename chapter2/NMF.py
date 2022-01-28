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

mu = 1e-6


def grads(M, W, H):
    R = W @ H - M

    return R @ H.T + penalty(W, mu) * lam, W.T @ R + penalty(H, mu) * lam


def penalty(M, mu):
    return np.where(M >= mu, 0, np.min(M - mu, 0))


def upd(M, W, H, lr):
    dW, dH = grads(M, W, H)

    W -= lr * dW
    H -= lr * dH


def report(M, W, H):
    print(np.linalg.norm(M - W @ H), W.min(), H.min(), (W < 0).sum, (H < 0).sum())


W = np.abs(np.random.normal(scale=0.01, size=(m, d)))
H = np.abs(np.random.normal(scale=0.01, size=(d, n)))

report(vectors_tfidf, W, H)

upd(vectors_tfidf, W, H, lr)
print('-'*30)
report(vectors_tfidf, W, H)

for i in range(50):
    upd(vectors_tfidf, W, H, lr)
    if i % 30 == 0:
        print('-'*30)
        report(vectors_tfidf, W, H)

print(show_topics(H))
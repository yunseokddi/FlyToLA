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
# Is the .CEL file format available from somewhere?
#
# Rych
#
#
# Seems to be, barring evidence to the contrary, that Koresh was simply
# another deranged fanatic who thought it neccessary to take a whole bunch of
# folks with him, children and all, to satisfy his delusional mania. Jim
# Jones, circa 1993.
#
#
# Nope - fruitcakes like Koresh have been demonstrating such evil corruption
# for centuries.
#
#  >In article <1993Apr19.020359.26996@sq.sq.com>, msb@sq.sq.com (Mark Brader)
#
# MB>                                                             So the
# MB> 1970 figure seems unlikely to actually be anything but a perijove.
#
# JG>Sorry, _perijoves_...I'm not used to talking this language.
#
# Couldn't we just say periapsis or apoapsis?


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

U, s, Vh = linalg.svd(vectors, full_matrices=False)
# print(U.shape, s.shape, Vh.shape)
# (2034, 2034) (2034,) (2034, 26576)
# print(s[:10])
plt.plot(s)
plt.show()
plt.plot(s[:10])
plt.show()

num_top_words = 8


def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words - 1: -1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]

result = show_topics(Vh[:10])
print(result)
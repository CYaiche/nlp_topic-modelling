import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# word count
import gensim
import gensim.corpora as corpora
from gensim.models import Word2Vec

from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import TruncatedSVD





## Preprocessing and dimension reduction : TF-IDF and LSA
def getTDF(corpus, dic ):
    """ retrun Term Document Frequency"""
    return  [ dic.doc2bow(text) for text in corpus]

def tfidf_to_xsparse(tfidf_in, nb_lines, nb_cols):
    """Convert the TF-IDF corpus into a sparse matrix"""
    rows, cols, data = [], [], []
    for i, doc in enumerate(tfidf_in):
        for j, value in doc:
            rows.append(i)
            cols.append(j)
            data.append(value)
    return csr_matrix((data, (rows, cols)), shape=(nb_lines, nb_cols))

def tfidf_lsa_preprocessing(X_train, X_test):
    """ apply tf-idf and lsa to bag-of-words with train-test separation
    """
    id2word = corpora.Dictionary(X_train)
    corpus = getTDF(X_train, id2word)
    tfidf = gensim.models.TfidfModel(corpus)

    tfidf_c_train = tfidf[getTDF(X_train, id2word)]
    tfidf_c_t_test = tfidf[getTDF(X_test, id2word)]
    # TRAIN
    X_sparse = tfidf_to_xsparse(tfidf_c_train, len(tfidf_c_train), len(id2word))
    # train LSA on train set
    svd = TruncatedSVD(n_components=1100, n_iter=7, random_state=33)
    X_train_svd = svd.fit_transform(X_sparse)

    # apply PCA on test set
    X_sparse_test = tfidf_to_xsparse(tfidf_c_t_test, len(tfidf_c_t_test), len(id2word))
    X_svd_test = svd.transform(X_sparse_test)

    return X_train_svd, X_svd_test



def tm_get_working_config():
    """ get boolean value for running on colab or not, return working directory"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        output_dir = "/content/drive/MyDrive/OpenClassroom/"
        # !pip install bertopic
        IN_COLAB = True
    except:
        IN_COLAB = False
        output_dir = "./output/"

    return IN_COLAB, output_dir

def tm_append_title_body_test_train(X_title_train, X_body_train, X_title_test, X_body_test) : 
    X_train = [
        np.append(X_title_train[i], X_body_train[i]) for i in range(len(X_title_train))
    ]
    X_test = [np.append(X_title_test[i], X_body_test[i]) for i in range(len(X_title_test))]
    return X_train, X_test

def tm_load_train_test_set(output_dir, option="append"):
    if option =="append" : 
        X_title_train = np.load(f"{output_dir}X_title_train.npy", allow_pickle=True)
        X_body_train = np.load(f"{output_dir}X_body_train.npy", allow_pickle=True)

        X_title_test = np.load(f"{output_dir}X_title_test.npy", allow_pickle=True)
        X_body_test = np.load(f"{output_dir}X_body_test.npy", allow_pickle=True)

        y_train = np.load(f"{output_dir}y_train.npy", allow_pickle=True)
        y_test = np.load(f"{output_dir}y_test.npy", allow_pickle=True)
        
        X_train, X_test = tm_append_title_body_test_train(X_title_train, X_body_train, X_title_test, X_body_test)
    return  X_train, X_test, y_train, y_test



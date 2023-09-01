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

from sklearn.metrics import jaccard_score, average_precision_score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


## Preprocessing and dimension reduction : TF-IDF and LSA
def getTDF(corpus, dic ):
    """ return Term Document Frequency"""
    return  [ dic.doc2bow(text) for text in corpus]

def tfidf_to_xsparse(tfidf_in, nb_lines, nb_cols):
    """ Convert the TF-IDF corpus into a sparse matrix"""
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
    """ returns boolean value "if running on colab" and working directory"""
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
    """ concatenates title and body posts both for train and test set
    """
    X_train = [
        np.append(X_title_train[i], X_body_train[i]) for i in range(len(X_title_train))
    ]
    X_test = [np.append(X_title_test[i], X_body_test[i]) for i in range(len(X_title_test))]
    return X_train, X_test

def tm_load_train_test_set(output_dir, option="append"):
    """ returns train test set
    """
    if option =="raw_corpus" : 
        
        X_corpus_train = np.load(f"{output_dir}X_corpus_train.npy", allow_pickle=True)
        X_corpus_test = np.load(f"{output_dir}X_corpus_test.npy", allow_pickle=True)
        
        return X_corpus_train, X_corpus_test

    elif option =="append" : 
        X_title_train = np.load(f"{output_dir}X_title_train.npy", allow_pickle=True)
        X_body_train = np.load(f"{output_dir}X_body_train.npy", allow_pickle=True)

        X_title_test = np.load(f"{output_dir}X_title_test.npy", allow_pickle=True)
        X_body_test = np.load(f"{output_dir}X_body_test.npy", allow_pickle=True)

        y_train = np.load(f"{output_dir}y_train.npy", allow_pickle=True)
        y_test = np.load(f"{output_dir}y_test.npy", allow_pickle=True)
        
        X_train, X_test = tm_append_title_body_test_train(X_title_train, X_body_train, X_title_test, X_body_test)
    
    return  X_train, X_test, y_train, y_test

def tm_get_label_list(output_dir):
    return np.load(f"{output_dir}/label_list.npy", allow_pickle=True)

def tm_get_subset(X_train, y_train) : 
    """ to work on a subset of train test set, here 25 % """
    size_train = len(X_train) // 4 
    return X_train[:size_train],  y_train[:size_train]

def tm_get_subsetX(X_train) : 
    """ to work on a subset of train test set, here 25 % """
    size_train = len(X_train) // 4 
    return X_train[:size_train]

def tm_multilabel_binarizer(y_train, y_test) : 
    " multi-labels to hot-encoding like target"
    mlb = MultiLabelBinarizer()
    y_train_b = mlb.fit_transform(y_train)
    y_test_b = mlb.transform(y_test)
    return y_train_b, y_test_b


# ********************** Thresholds ******************* #

def tm_test_threshold(y_train_b, y_pred) : 
    """ test different threshold to convert probability output into class membership decision """
    config = {"threshold" : [],
                "precision" : [], 
                "jaccard" : []
                }
    thr = np.arange(-9,0.2,0.2)
    test_thr = 10**thr
    for t in test_thr : 
        y_pred_t = (y_pred > t ).astype(np.float32)
        prec      = average_precision_score(y_train_b, y_pred_t, average='micro')
        jacc = jaccard_score(y_train_b, y_pred_t, average='micro')
        
        config["threshold"].append(t)
        config["precision"].append(prec)
        config["jaccard"].append(jacc)
    return pd.DataFrame(config)

def tm_plot_threshold_test(threshold_test): 
    """ plots different threshold and precision """
    plt.plot(threshold_test["threshold"], threshold_test["precision"],label="precsion")
    plt.plot(threshold_test["threshold"], threshold_test["jaccard"],label="jaccard_score")
    plt.xlabel("Decision threshold")
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(y_test, y_pred, label_list, NB_LABEL=3):
    fig, axs = plt.subplots(1, NB_LABEL, figsize=(NB_LABEL * 10, 10))
    for i in range(NB_LABEL):
        rdm_label = i * 5
        cm = multilabel_confusion_matrix(y_test, y_pred)[rdm_label]

        display_labels = ["others", label_list[rdm_label]]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )
        disp.plot(ax=axs[i], colorbar=False)
        disp.figure_.tight_layout()
    plt.rc('font', size=30)
    plt.show()
    
    
    
    # plt.rc('font', size=14)
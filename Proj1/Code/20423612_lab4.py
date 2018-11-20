# author: TSE, Ho Nam
# student_id: 20423612
import numpy as np
from scipy import sparse
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

# stop_words = set(stopwords.words('english') + list(string.punctuation))
stop_words = []

def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g. 
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    # YOUR CODE HERE
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens

def get_bagofwords(data, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param vocab_dict: a dict from words to indices, type: dict
    return a word (sparse) matrix, type: scipy.sparse.csr_matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
    '''
    data_matrix = sparse.lil_matrix((len(data), len(vocab_dict)))
    # YOUR CODE HERE
    for i, doc in enumerate(data):
        for word in doc:
            word_idx = vocab_dict.get(word, -1)
            if word_idx != -1:
                data_matrix[i, word_idx] += 1
    data_matrix = data_matrix.tocsr()
    return data_matrix

def read_data(file_name, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    data_matrix = get_bagofwords(df['words'], vocab_dict)
    return df['id'], df['label'], data_matrix, vocab

def normalize(P):
    """
    normalize P to make sure the sum of every row equals to 1
    e.g.
    Input: [1,2,1,2,4]
    Output: [0.1,0.2,0.1,0.2,0.4] (without laplace smoothing) or [0.1333,0.2,0.1333,0.2,0.3333] (with laplace smoothing)
    """
    # YOUR CODE HERE
    alpha = 1 # change this for Laplace smoothing
    K = P.shape[0]
    norm = np.sum(P, axis=0, keepdims=True)
    return (P+alpha) / (norm+alpha*K)
    
def train_NB(data_label, data_matrix):
    '''
    :param data_label: [N], type: list
    :param data_matrix: [N(document_number), V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    return the P(y) (an M array), P(x|y) (a V*M matrix)
    '''
    N = data_matrix.shape[0]
    K = max(data_label) # labels begin with 1
    # YOUR CODE HERE
    data_delta = np.zeros((N, K))
    for i, l in enumerate(data_label):
        data_delta[i, l-1] = 1
    P_y = normalize(np.sum(data_delta, axis=0))
    P_xy = normalize(data_matrix.T.dot(data_delta))

    return P_y, P_xy
    
def predict_NB(data_matrix, P_y, P_xy):
    '''
    :param data_matrix: [N(document_number), V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    :param P_y: [M(label number)], type: np.ndarray
    :param P_xy: [V, M], type: np.ndarray
    return data_pre (a N array)
    '''
    # compute the label probabilities using the P(y) and P(x|y) according to the naive Bayes algorithm
    # YOUR CODE HERE
    log_P_y = np.expand_dims(np.log(P_y), axis=0)
    log_P_xy = np.log(P_xy)
    log_P_dy = data_matrix.dot(log_P_xy)
    log_P = log_P_y + log_P_dy
    # get labels for every document by choosing the maximum probability
    # YOUR CODE HERE
    P = np.exp(log_P)
    print(P)
    e = P/np.sum(P)
    print('Prob:', P/np.sum(P))
    return np.argmax(log_P, axis=1) + 1
    

def evaluate(y_true, y_pre):
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    return acc, precision, recall, f1

if __name__ == '__main__':
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv")
    print("Vocabulary Size:", len(vocab), vocab)
    print("Training Set Size:", len(train_id_list))
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", vocab)
    print("Test Set Size:", len(test_id_list))

    P_y, P_xy = train_NB(train_data_label, train_data_matrix)
    print('P_y:', P_y)
    out = []
    for i, v in enumerate(vocab):
        out.append((v, P_xy[i]))
    for p in sorted(out):
        w, [p1, p2] = p
        print('\\text{'+w+'} & ' + str(p1) + ' & ' + str(p2) + '\\\\\\midrule')

    # train_data_pre = predict_NB(train_data_matrix, P_y, P_xy)
    # acc, precision, recall, f1 = evaluate(train_data_label, train_data_pre)
    # print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    test_data_pre = predict_NB(test_data_matrix, P_y, P_xy)

    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list
    sub_df["pred"] = test_data_pre
    sub_df.to_csv("20423612_lab4.csv", index=False)
    
    # grt_label = pd.read_csv("data/answer.csv")
    # acc, precision, recall, f1 = evaluate(grt_label["label"], test_data_pre)
    # print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

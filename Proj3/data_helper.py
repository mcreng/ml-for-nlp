import math
import json
import os
import numpy as np
import pandas as pd

def build_input_data(sentences, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    
    Args:
        sentences (pd.Dataframe): date frame of raw sentences
        vocabulary (dict): Dictionary of vocab key pairs

    Returns:
        (list[int]): index list
        (iterator): input word ids
        (iterator): target word ids
        int: number of sentences
    """
    unknown_token_id = vocabulary["<unk>"]
    vocab = vocabulary.keys()

    def sent2idx(sentence):
        """
        Converts words into ids from vocab.json

        Args:
            sentence (str): Raw string of sentence

        Returns:
            sentence (list[int]): List of word ids
        """
        sentence = sentence.split(' ')
        sentence = [vocabulary[word] if word in vocab else unknown_token_id for word in sentence]
        return sentence

    sentences = sentences.applymap(sent2idx)
    
    sentences['target'] = (sentences['sentence'].map(lambda a: a[1:]) + sentences['label']).map(lambda a: np.array([np.array([a]).T]))
    sentences['sentence'] = sentences['sentence'].map(lambda a: np.array([a]))

    return sentences.index.tolist(), iter(zip(sentences['sentence'].tolist(), sentences['target'].tolist())), len(sentences)


def load_data(data_path, file):
    """
    Load data for training.

    Args:
        data_path (str): Data path
        file (str): filename

    Returns:
        (list[int]): index list
        (iterator): training data sets of (x, y)
        (iterator): validation data sets of (x, y)
        num_training_data (int): number of training data
        num_valid_data (int): number of validation data
        vocab_size (int): size of vocabulary
    """
    # get the data paths
    path = os.path.join(data_path, "{}.csv".format(file))
    vocab_path = os.path.join(data_path, "vocab.json")

    # build vocabulary from training data
    vocabulary = json.load(open(vocab_path))
    vocab_size = len(vocabulary)

    # get input data
    idx, data, num_data = build_input_data(pd.read_csv(path, index_col=0), vocabulary)

    return idx, data, num_data, vocab_size

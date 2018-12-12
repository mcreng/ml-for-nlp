from __future__ import print_function
from math import ceil
import argparse
import json
import os
import time
import itertools
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Dropout, TimeDistributed, Conv1D, BatchNormalization, Reshape, Activation, LSTM, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
from data_helper import load_data, build_input_data
from scorer import scoring
from utils import make_submission
from datetime import datetime

from tcn_layer import TCN

def build_model(embedding_dim, hidden_size, drop, vocabulary_size):
    inputs = Input(shape=(None,), dtype='int32')

    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=None)(inputs)

    tconv1 = TCN(nb_filters=hidden_size, dilations=[1, 2, 4, 8], return_sequences=True, dropout_rate=drop, name='tcn1')(embedding)
    lstm1 = LSTM(units=hidden_size, dropout=drop, recurrent_dropout=drop, return_sequences=True)(embedding)
    outputs = Dense(units=vocabulary_size, activation='softmax')(concatenate([tconv1, lstm1, embedding]))

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    print(model.summary())
    return model


def data_gen(data, predicting=False):
    """
    Training/Testing data generator function. Zero paddings were used so all data in a batch has the same length.

    Args:
        data (iterator): data iterator from data_helper.py
        predicting (boolean): whether data is used to fit or predict
    
    Returns:
        if predicting,
        d (np.ndarray): list of indices indicating the last entry of each sentence
        x (np.ndarray): list of input word ids

        if fitting,
        x (np.ndarray): list of input word ids
        y (np.ndarray): list of target word ids
    """
    if predicting:
        while True:
            x, _ = zip(*data)
            maxl = max(map(lambda x: x.shape[1], x))
            d = np.array([len(el[0]) for el in x])
            x = np.array([np.pad(el[0], (0, maxl - len(el[0])), 'constant', constant_values=(0)) for el in x])
            yield d, x
    else:
        while True:
            x, y = zip(*itertools.islice(data, opt.batch_size))
            maxl = max(map(lambda x: x.shape[1], x))
            x = np.array([np.pad(el[0], (0, maxl - len(el[0])), 'constant', constant_values=(0)) for el in x])
            y = np.array([np.array([np.pad(el[0].T[0], (0, maxl - len(el.T[0])), 'constant', constant_values=(0))]).T for el in y])
            yield x, y

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    if opt.mode == "train":
        st = time.time()
        print('Loading data')
        _, train, num_training_data, vocabulary_size = load_data('data', 'train')
        _, valid, num_valid_data, _ = load_data('data', 'valid')

        print(num_training_data)
        print(num_valid_data)

        print('Vocab Size', vocabulary_size)

        model = build_model(opt.embedding_dim, opt.hidden_size, opt.drop, vocabulary_size)
        print("Training Model...")

        es = EarlyStopping(restore_best_weights=True, patience=3)

        curtime = str(datetime.now())
        os.mkdir('./logs/'+curtime)
        tb = TensorBoard(log_dir='./logs/'+curtime, histogram_freq=0, write_grads=True, write_graph=True, write_images=True)

        history = model.fit_generator(data_gen(train), steps_per_epoch=ceil(num_training_data/opt.batch_size),
                            epochs=opt.epochs, verbose=1,
                            validation_data=data_gen(valid),
                            validation_steps=ceil(num_valid_data/opt.batch_size),
                            callbacks=[es, tb])
        model.save(opt.saved_model)
        print("Training cost time: ", time.time() - st)

    else:
        print('Loading data')
        idx, test, num_testing_data, _ = load_data('data', opt.input)

        def exhaust_data_gen():
            try:
                yield from data_gen(test, predicting=True)
            except:
                pass
        
        pos, test = zip(*list(exhaust_data_gen()))

        pos = pos[0]
        test = test[0]

        model = load_model(opt.saved_model)

        prob = model.predict(test, batch_size=opt.batch_size, verbose=1) # cannot use predict_generator on varying output length...
        prob = np.array([prob[b, pos[b]-1, :] for b in range(prob.shape[0])])
        prob = dict(zip(idx, prob))

        sub_file = make_submission(prob, opt.student_id, opt.input)
        if opt.score:
            scoring(sub_file, os.path.join("data"), type="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default="train", choices=["train", "test"],
                        help="Train or test mode")
    parser.add_argument("-saved_model", type=str, default="model.h5",
                        help="saved model path")
    parser.add_argument("-input", type=str, default='valid', choices=['valid', 'test'],
                        help="Input path for generating submission")
    parser.add_argument("-debug", action="store_true",
                        help="Use validation data as training data if it is true")
    parser.add_argument("-score", action="store_true",
                        help="Report score if it is")
    parser.add_argument("-student_id", default=None, required=True,
                        help="Student id number is compulsory!")

    parser.add_argument("-epochs", type=int, default=1,
                        help="training epoch num")
    parser.add_argument("-batch_size", type=int, default=32,
                        help="training batch size")
    parser.add_argument("-embedding_dim", type=int, default=100,
                        help="word embedding dimension")
    parser.add_argument("-hidden_size", type=int, default=500,
                        help="rnn hidden size")
    parser.add_argument("-drop", type=float, default=0.5,
                        help="dropout")
    parser.add_argument("-gpu", type=str, default="",
                        help="dropout")
    opt = parser.parse_args()
    main(opt)

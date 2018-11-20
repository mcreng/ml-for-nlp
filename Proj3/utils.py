# You should NOT change this file
import math
import numpy as np
from keras.callbacks import Callback


class TestCallback(Callback):
    """
    Calculate Perplexity
    """
    def __init__(self, test_data, model):
        super(TestCallback, self).__init__()
        self.test_data = test_data
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        x_probs = self.model.predict(x)
        ppl = self.evaluate_batch_ppl(x_probs,y)
        print('\nValidation Set Perplexity: {0:.2f} \n'.format(ppl))

    def evaluate_ppl(self, x, y):
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)
        return np.exp(np.mean(-np.log(np.diag(x[:, y]))))

    def evaluate_batch_ppl(self, x, y):
        eval_batch_size = 8
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)
        ppl = 0.0
        for i in range(math.ceil(len(x)/eval_batch_size)):
            batch_x = x[i*eval_batch_size:(i+1)*eval_batch_size,:]
            batch_y = y[i*eval_batch_size:(i+1)*eval_batch_size]
            ppl += np.sum(np.log(np.diag(batch_x[:, batch_y])))
        return np.exp(-ppl/x.shape[0])


def make_submission(predict_dict, stuid, input_path):
    assert stuid.isdigit(), "input valid student id!"
    if "valid" in input_path:
        filename = stuid + "_valid_result.csv"
    else:
        filename = stuid + "_result.csv"
    fout = open(filename, "w")
    for id_ in sorted(predict_dict.keys()):
        fout.write(str(id_) + ",")
        probs = " ".join(["%.8f" % t for t in predict_dict[id_]])
        fout.write(probs + "\n")
    fout.close()
    return filename

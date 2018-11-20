# You should NOT change this file

import argparse
import json
import os
import numpy as np
from sklearn.metrics import log_loss


def scoring(sub_file, data_path, type="valid"):
    id_probs = []
    with open(sub_file) as f:
        for line in f:
            id_, probs = line.strip().split(",")
            id_ = int(id_)
            probs = [float(t) for t in probs.split()]
            id_probs.append((id_, probs))
    id_probs = sorted(id_probs, key=lambda x: x[0])
    y_prob = np.array([t[1] for t in id_probs])

    vocab = json.load(open(os.path.join(data_path, "vocab.json")))

    y_true = []
    with open(os.path.join(data_path, "%s.csv" % type), "r") as fin:
        fin.readline()
        for line in fin:
            *_, grt_last_token = line.strip().split(",")
            idx = vocab[grt_last_token] if grt_last_token in vocab else vocab["<unk>"]
            y_true.append(idx)
    y_true = np.array(y_true)

    assert y_prob.shape[0] == y_true.shape[0] and y_prob.shape[1] == len(vocab), \
        "Wrong submission format, check your valid row/col number"

    # y_pred = np.argmax(y_prob, axis=1)
    # print("accuracy: ", accuracy_score(y_true, y_pred))
    final_score = log_loss(y_true, y_prob, labels=list(range(len(vocab))))
    print("Your score on validation data: %.4f" % final_score)

    print("=" * 50)
    if final_score > 2.6795:
        print("                          <- You're here (0pts)")
    print("[60  / 100] Baseline: 2.6795 ")
    if 1.9034 < np.round(final_score, 4) <= 2.6795:
        print("                          <- You're here (60pts)")
    print("[80  / 100] Baseline: 1.9034 ")
    if 1.7830 < np.round(final_score, 4) <= 1.9034:
        print("                          <- You're here (80pts)")
    print("[90  / 100] Baseline: 1.7830 ")
    if 1.7345 < np.round(final_score, 4) <= 1.7830:
        print("                          <- You're here (90pts)")
    print("[100 / 100] Baseline: 1.7345 ")
    if 1.6737 < np.round(final_score, 4) <= 1.7345:
        print("                          <- You're here (100pts)")
    print("[  BONUS  ] Baseline: 1.6737 ")
    if np.round(final_score, 4) <= 1.6737:
        print("                          <- You may get a bonus. Good Job!")
    print("\nATTENTION! This submission/score is generated from validation data, \n"
          "you may get 0pts if you submit it on Canvas!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-submission", type=str,
                        help="submission file")
    parser.add_argument("-data_path", type=str, default=os.path.join("data"))
    parser.add_argument("-type", type=str, default="valid", choices=["valid"])
    opt = parser.parse_args()
    scoring(opt.submission, opt.data_path, opt.type)
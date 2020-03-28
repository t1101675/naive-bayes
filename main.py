# codeing=utf-8

import os
import re
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tokenization import Tokenizer


tokenizer = Tokenizer()

train_file = "trec06p/label/index"
test_file = "trec06p/label/small_index"
data_dir = "trec06p/data"
num_labels = 2
M = 2
alpha = 0.1
seed = 888

random.seed(seed)

def load_file(index_file, shuffle=True):
    with open(index_file, "r") as f:
        indices = f.readlines()
    
    if shuffle:
        random.shuffle(indices)

    # indices = indices[0:100]

    all_mails = []
    all_labels = []

    for line in tqdm(indices, desc="Loading files"):
        label, path = line.strip().split(" ")
        label = 1 if label == "ham" else 0
        path = path.strip().split("/")
        path = os.path.join(data_dir, path[2], path[3])
        with open(path, "r", encoding="UTF-8") as f:
            try:
                all_mails.append(f.read())
            except UnicodeDecodeError:
                continue
            all_labels.append(label)

    return all_mails, all_labels


def clean(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.replace("&nbsp", "")
    
    return text


def preprocess_one_file(mail):
    mail = mail.split("\n\n")
    head = mail[0]
    tmp = " ".join(mail[1:])
    tmp = tmp.split("\n")
    body = []
    for line in tmp:
        if line:
            body.append(line)

    body = " ".join(body)
    body = clean(body)

    body = tokenizer.tokenize(body)

    return {
        "head": head,
        "body": body
    }


def build_vocab(mails):
    S = []
    for mail in tqdm(mails, desc="Building Vocab"):
        body = mail["body"]
        S.extend(body)
    S = list(set(S))
    print("Vocab size: {}".format(len(S)))
    # print(S)
    # exit(0)
    id_2_w = ["[UNK]"] + S
    w_2_id = {w: i for i, w in enumerate(id_2_w)}

    return id_2_w, w_2_id


def preprocess(mails, do_train=True):
    new_mails = []
    desc = ("Train" if do_train else "Test") + " Preprocessing"
    
    for i in tqdm(range(len(mails)), desc=desc):
        new_mails.append(preprocess_one_file(mails[i]))

    if do_train:
        id_2_w, w_2_id = build_vocab(new_mails)
        bow_vecs = np.zeros([len(new_mails), len(id_2_w)])
        for i, mail in enumerate(tqdm(new_mails, desc="Building Bag of Words")):
            body = mail["body"]
            for w in body:
                bow_vecs[i][w_2_id[w]] += 1

        return id_2_w, w_2_id, bow_vecs
    else:
        return new_mails


def train(id_2_w, w_2_id, bow_vecs, all_train_labels):
    probs = np.zeros([num_labels, len(id_2_w)])
    all_counts = np.zeros(num_labels)
    pos_num = 0
    for vec, label in zip(tqdm(bow_vecs), all_train_labels):
        probs[label] += vec
        all_counts[label] += np.sum(vec)
        pos_num += label

    for label in range(num_labels):
        t1 = probs[label] + alpha
        t2 = all_counts[label] + M * alpha
        t3 = t1 / t2

        probs[label] = np.log(t3)

    # print(probs)    

    pos_prob = np.log(pos_num / len(all_train_labels))
    neg_prob = np.log(1 - pos_num / len(all_train_labels))

    # print(pos_num)
    # print(len(all_train_labels))

    return probs, pos_prob, neg_prob
    

def test(id_2_w, w_2_id, probs, pos_prob, neg_prob, mails):
    preds = []
    for mail in tqdm(mails, desc="Testing"):
        # print(mail)
        body = mail["body"]
        word_ids = [w_2_id[w] if w in w_2_id else w_2_id["[UNK]"] for w in body]
        neg = sum([probs[0][index] for index in word_ids]) + neg_prob
        pos = sum([probs[1][index] for index in word_ids]) + pos_prob
        # print(neg, pos)
        if pos > neg:
            preds.append(1)
        else:
            preds.append(0)

    return preds


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    all_mails, all_labels = load_file(train_file, shuffle=True)
    L = len(all_labels) // 5
    results = []
    for e in range(5):
        print("Test #{}:".format(e))
        lo = e * L
        hi = lo + L
        all_train_mails = all_mails[0:lo] + all_mails[hi:]
        all_train_labels = all_labels[0:lo] + all_labels[hi:]
        all_test_mails = all_mails[lo:hi]
        all_test_labels = all_labels[lo:hi]

        id_2_w, w_2_id, bow_vecs = preprocess(all_train_mails, do_train=True)    
        probs, pos_prob, neg_prob = train(id_2_w, w_2_id, bow_vecs, all_train_labels)
        all_test_mails = preprocess(all_test_mails, do_train=False)
        preds = test(id_2_w, w_2_id, probs, pos_prob, neg_prob, all_test_mails)
        result = evaluate(all_test_labels, preds)

        results.append(result)

    avg_result = {key: np.mean([res[key] for res in results]) for key in results[0].keys()}

    print(avg_result)

if __name__ == "__main__":
    main()

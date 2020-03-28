# codeing=utf-8

import os
import re
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tokenization import Tokenizer


tokenizer = Tokenizer()

input_file = "trec06p/label/index"
data_dir = "trec06p/data"
num_labels = 2
M = 2
alpha = 0.1
seed = 888

random.seed(seed)

cand_nums = [21, 24, 21, 6]
features_num = len(cand_nums)
feature_weights = [5, 1, 1, 1]

from_features = ['[UNK]', 'hotmail', 'lingo', 'gmail', 'yahoo', 'aol', '0451', 'iname', 'singnet', 'www.loveinfashion', 'o-himesama',
                 'aries.livedoor', 'oh-oku', 'msn', 'paypal', 'tc.fluke', 'ey', 'specialdevices', 'buta-gori', 'plan9.bell-labs', 'halcyon']
from_features_map = {f: i for i, f in enumerate(from_features)}

mailer_features = ['[UNK]', 'Microsoft', 'Mozilla', 'The', 'QUALCOMM', 'Internet', 'Windows', 'AOL', 'ELM', 'Apple', 'Pegasus',
                   'dtmail', 'Sylpheed', 'Claris', 'exmh', 'Ximian', 'eGroups', 'Pine', 'uPortal', 'aspNetEmail', 'DMailWeb']
mailer_features_map = {f: i for i, f in enumerate(mailer_features)}

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

f = {}

def get_features(head):
    features = np.zeros(features_num, dtype=np.int32)
    lines = head.strip().split("\n")
    from_pattern = re.compile(r".*@(.*)\..*$")
    time_pattern = re.compile(r".* (\d+):(\d+):(\d+) .*")
    mailer_pattern = re.compile(r"X-Mailer: ([a-zA-Z]*) ")
    priority_pattern = re.compile(r"X-Priority: (\d)")

    for line in lines:
        if "From: " in line:
            m = from_pattern.match(line)
            if m:
                if m.group(1):
                    x = m.group(1)
                    features[0] = from_features_map[x] if x in from_features_map else from_features_map["[UNK]"]
        
        if "Date: " in line:
            m = time_pattern.match(line)
            if m:
                if m.group(1):
                    x = m.group(1)
                    features[1] = int(x)

        if "X-Mailer: " in line:
            m = mailer_pattern.match(line)
            if m:
                if m.group(1):
                    x = m.group(1)
                    # print(x)
                    # f[x] = f[x] + 1 if x in f else 1
                    features[2] = mailer_features_map[x] if x in mailer_features_map else mailer_features_map["[UNK]"]

        if "X-Priority: " in line:
            m = priority_pattern.match(line)
            if m:
                if m.group(1):
                    x = m.group(1)
                    # f[x] = f[x] + 1 if x in f else 1
                    features[3] = int(x)

    return features


def preprocess_one_file(mail):
    mail = mail.split("\n\n")
    head = mail[0]
    body = []
    tmp = " ".join(mail[1:])
    tmp = tmp.split("\n")
    for line in tmp:
        if line:
            body.append(line)

    body = " ".join(body)
    body = clean(body)

    body = tokenizer.tokenize(body)

    features = get_features(head)

    return {
        "head": head,
        "body": body,
        "features": features
    }


def build_vocab(mails):
    S = []
    for mail in tqdm(mails, desc="Building Vocab"):
        body = mail["body"]
        S.extend(body)
    S = list(set(S))
    print("Vocab size: {}".format(len(S)))
    id_2_w = ["[UNK]"] + S
    w_2_id = {w: i for i, w in enumerate(id_2_w)}

    return id_2_w, w_2_id


def cal_all_features(mails):
    all_features = np.zeros([len(mails), features_num], dtype=np.int32)
    for i, mail in enumerate(mails):
        all_features[i] = mail["features"]

    return all_features


def preprocess(mails, do_train=True):
    new_mails = []
    desc = ("Train" if do_train else "Test") + " Preprocessing"

    for i in tqdm(range(len(mails)), desc=desc):
        new_mails.append(preprocess_one_file(mails[i]))

    s = sorted(f.items(), key=lambda x: x[1], reverse=True)
    # print(s)
    # print([x[0] for x in s[0:20]])
    # exit(0)


    if do_train:
        id_2_w, w_2_id = build_vocab(new_mails)
        bow_vecs = np.zeros([len(new_mails), len(id_2_w)])
        all_features = cal_all_features(new_mails)
        

        for i, mail in enumerate(tqdm(new_mails, desc="Building Bag of Words")):
            body = mail["body"]
            for w in body:
                bow_vecs[i][w_2_id[w]] += 1

        return id_2_w, w_2_id, bow_vecs, all_features
    else:
        return new_mails


def train(id_2_w, w_2_id, bow_vecs, all_features, all_train_labels):
    probs = np.zeros([num_labels, len(id_2_w)])
    all_counts = np.zeros(num_labels)
    feature_probs = [np.zeros([num_labels, cand_num]) for cand_num in cand_nums]
    pos_num = 0
    for vec, features, label in zip(tqdm(bow_vecs), all_features, all_train_labels):
        probs[label] += vec
        all_counts[label] += np.sum(vec)
        pos_num += label

        for i, f in enumerate(features):
            feature_probs[i][label][f] += 1

    for label in range(num_labels):
        probs[label] = np.log((probs[label] + alpha) / (all_counts[label] + M * alpha))

        for i, cand_num in enumerate(cand_nums):
            feature_probs[i][label] = np.log(
                (feature_probs[i][label] + alpha) / (all_counts[label] + cand_num * alpha))


    pos_prob = np.log(pos_num / len(all_train_labels))
    neg_prob = np.log(1 - pos_num / len(all_train_labels))

    return probs, feature_probs, pos_prob, neg_prob
    

def test(id_2_w, w_2_id, probs, feature_probs, pos_prob, neg_prob, mails):
    preds = []
    for mail in tqdm(mails, desc="Testing"):
        body = mail["body"]
        features = mail["features"]
        word_ids = [w_2_id[w] if w in w_2_id else w_2_id["[UNK]"] for w in body]
        neg = sum([probs[0][index] for index in word_ids]) + neg_prob
        pos = sum([probs[1][index] for index in word_ids]) + pos_prob
        for feature, feature_prob, feature_weight in zip(features, feature_probs, feature_weights):
            neg += feature_weight * feature_prob[0][feature]
            pos += feature_weight * feature_prob[1][feature]

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
    all_mails, all_labels = load_file(input_file, shuffle=True)
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

        id_2_w, w_2_id, bow_vecs, all_train_features = preprocess(all_train_mails, do_train=True)    
        probs, feature_probs, pos_prob, neg_prob = train(id_2_w, w_2_id, bow_vecs, all_train_features, all_train_labels)
        all_test_mails = preprocess(all_test_mails, do_train=False)
        preds = test(id_2_w, w_2_id, probs, feature_probs, pos_prob, neg_prob, all_test_mails)
        result = evaluate(all_test_labels, preds)

        results.append(result)

        print(result)
        exit(0)

    avg_result = {key: np.mean([res[key] for res in results]) for key in results[0].keys()}

    print(avg_result)

if __name__ == "__main__":
    main()

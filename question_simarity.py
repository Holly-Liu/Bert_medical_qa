import json
import time
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_iterator, get_time_dif, build_dataset_qa, build_dev_generator, build_dataset_cons_learning, \
    build_dataset_simarity
from my_log import Logger

from sklearn.cluster import KMeans
from text_len import TextLenAnalyser




def model_compute():
    model_name = "bert_mean_pool"  # bert
    x = import_module('models.' + model_name)
    config = x.Config("")
    config.train_method = "simarity"
    config.batch_size = 32
    model = x.Model(config).to(config.device)
    ques = build_dataset_simarity(config)
    ques_iter = build_iterator(ques, config, iter_type="train")
    que_id_all = []
    que_vec_all = []
    for i, train_batch in tqdm(enumerate(ques_iter)):
        model.eval()
        que_ids, que= train_batch
        que_out = model(que)
        que_vec = que_out.detach().cpu().numpy().tolist()
        que_id_all.extend(que_ids)
        que_vec_all.extend(que_vec)
    return que_id_all, que_vec_all



def show(n):
    with open("outputs/que_id_all.json", "r") as r:
        que_id_all = json.load(r)
    with open(f"outputs/k{n}_label.json", "r") as r:
        que_label_all = json.load(r)
    qa_label_map = {}
    for que_id, label in zip(que_id_all, que_label_all):
        qa_label_map[que_id] = label
    df_questions = pd.read_csv("/home/users/liuhongli/Models/datasets/cMedQA/questions.csv")
    tqdm.pandas(desc="applying")
    df_questions['label'] = df_questions['que_id'].progress_apply(lambda x: qa_label_map[x] if x in qa_label_map else -1)
    return df_questions

def train_and_cluster():
    ns = [1000, 2000, 3000, 4000, 5000, 7500, 10000]
    que_id_all, que_vec_all = model_compute()
    with open(f"outputs/que_id_all.json", 'w') as w:
        json.dump(que_id_all, w)
    for n in ns:
        print("kmeans " + str(n))
        kmeans = KMeans(n_clusters=n)
        res = kmeans.fit(que_vec_all)
        print("writing :" + str(n))
        with open(f"outputs/k{n}_label.json", 'w') as w:
            json.dump(res.labels_.tolist(), w)

if __name__ == '__main__':
    df_questions = show(10000)
    df_train = pd.read_csv('/home/users/liuhongli/Models/datasets/cMedQA/train_candidates.txt')
    train_ques = set(df_train['question_id'].to_list())
    df_dev = pd.read_csv('/home/users/liuhongli/Models/datasets/cMedQA/dev_candidates.txt')
    df_test = pd.read_csv('/home/users/liuhongli/Models/datasets/cMedQA/test_candidates.txt')
    dev_ques = set(df_dev['question_id'].to_list())
    test_ques = set(df_test['question_id'].to_list())
    df_questions['from_set'] = df_questions['que_id'].apply(lambda x: "train" if x in train_ques else "dev" if x in dev_ques else "test" if x in test_ques else "none")
    df_test_que = df_questions[df_questions['from_set'] == 'test'].rename(columns={"que_id": "test_que_id"})
    df_train_que = df_questions[df_questions['from_set'] == 'train'].rename(columns={"que_id": "train_que_id"})
    df_merge = pd.merge(df_test_que, df_train_que, on='label')
    df_group = df_merge.groupby("test_que_id")
    group = list(df_group)
    count = [0] * (2000 - len(df_group))
    for que_id, df_gp in group:
        count.append(len(df_gp))
    for que_id, df_gp in group[5:10]:
        test_question = df_questions[df_questions['que_id'] == que_id]['content'].to_list()
        similat_train_questions = df_gp["content_y"].to_list()
        print("test question: ")
        print(test_question)
        print("similar train question: ")
        print("\n".join(similat_train_questions))



    print()










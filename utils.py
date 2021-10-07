# coding: UTF-8
import random
from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from models.bert_medical_qa import Config

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def tokenizer_and_pad(sentence, tokenizer, pad_size):
    token = tokenizer.tokenize(sentence)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = tokenizer.convert_tokens_to_ids(token)
    pad_size = pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    return token_ids, mask, seq_len


def tokenize_df(df: pd.DataFrame, tokenizer,pad_size, token_columns):
    """
        根据给定columns 对 pd的指定列进行tokenize
    """
    if isinstance(token_columns, str):
        token_columns = [token_columns]
    for token_column in token_columns:
        tqdm.pandas(desc=f"{token_column}-tokenizing")
        df[f"{token_column}_tokens"] = df[token_column].progress_apply(lambda x: tokenizer_and_pad(x, tokenizer, pad_size))
    return df


def load_and_tokenize_cMedQA(config):
    question_path, answer_path, train_path, dev_path, test_path = get_paths_cMedQA(config.data_forder)
    df_questions = pd.read_csv(question_path) \
        .rename(columns={"content": "question"})[["que_id", "question"]]
    df_answers = pd.read_csv(answer_path) \
        .rename(columns={"content": "answer", "que_id": "ans_que_id"})[["ans_id", "answer", "ans_que_id"]]
    if config.debug_short_set != -1:
        df_questions = df_questions[:3000]
        df_answers = df_answers[:3000]

    df_questions = tokenize_df(df_questions, config.tokenizer, config.pad_size, "question")
    df_answers = tokenize_df(df_answers, config.tokenizer,config.pad_size, "answer")
    df_train = pd.read_csv(train_path).rename(columns={"question_id": "que_id"})
    df_dev = pd.read_csv(dev_path).rename(columns={"question_id": "que_id"})
    df_test = pd.read_csv(test_path).rename(columns={"question_id": "que_id"})
    return df_questions, df_answers, df_train, df_dev, df_test

def get_paths_cMedQA(forder_path):
    question_path = forder_path + '/questions.csv'
    answer_path = forder_path + '/answers.csv'
    train_path = forder_path + '/train_candidates.txt'
    dev_path = forder_path + "/dev_candidates.txt"
    test_path = forder_path + '/test_candidates.txt'
    return question_path, answer_path, train_path, dev_path, test_path

def get_paths_CHIP_STS(forder_path):
    train_path = forder_path + "/train_candidates.txt"
    dev_path = forder_path + "/dev_candidates.txt"
    test_path = forder_path + "/test_candidates.txt"
    return train_path, dev_path, test_path

def zip_dataframe(df: pd.DataFrame, columns=[]):
    df_list = [df[col].to_list() for col in columns]
    return list(zip(*df_list))


def build_dataset(config):
    if config.data_method == "cMedQA_origin":
        return build_dataset_cMedQA_origin(config)
    if config.data_method == "cMedQA_pos_ans_only":
        return build_dataset_cMedQA_pos_ans_only(config)
    if config.data_method == "cMedQA_que_only":
        return build_dataset_cMedQA_que_only(config)
    if config.data_method == "CHIP-STS_origin":
        return build_dataset_CHIPSTS_orgin(config)
    if config.data_method == "CHIP-STS_pos_only":
        return build_dataset_CHIPSTS_pos_only(config)
    if config.data_method == "unsup_cMedQA_dev_CHIPSTS":
        return build_dataset_unsup_cMedQA_dev_CHIPSTS(config)



def build_dataset_cMedQA_origin(config):
    start_time = time.time()
    df_questions, df_answers, df_train, df_dev, df_test = load_and_tokenize_cMedQA(config)
    print(f"Reading and tokenizing  cost time: {get_time_dif(start_time)}")
    df_pos_answers = df_answers.rename(columns={"answer_tokens": "pos_answer_tokens", "ans_id": "pos_ans_id"})
    df_neg_answers = df_answers.rename(columns={"answer_tokens": "neg_answer_tokens", "ans_id": "neg_ans_id"})
    df_train_join = df_train.merge(df_questions, on="que_id")\
                            .merge(df_pos_answers, on="pos_ans_id")\
                            .merge(df_neg_answers, on="neg_ans_id")[['question_tokens', 'pos_answer_tokens', "neg_answer_tokens"]]
    df_dev_join = df_dev.merge(df_questions, on="que_id")\
                        .merge(df_answers, on="ans_id")[["que_id", "question_tokens", "answer_tokens", "label"]]
    df_test_join = df_test.merge(df_questions, on="que_id")\
                          .merge(df_answers, on="ans_id")[["que_id", "question_tokens", "answer_tokens", "label"]]
    start_time = time.time()
    train = zip_dataframe(df_train_join, ["question_tokens", "pos_answer_tokens", "neg_answer_tokens"])
    dev = zip_dataframe(df_dev_join, ["que_id", "question_tokens", "answer_tokens", "label"])
    test = zip_dataframe(df_test_join, ["que_id", "question_tokens", "answer_tokens", "label"])
    print(f"Zip and transfer to list cost time: {get_time_dif(start_time)}")
    return train, dev, test


def build_dataset_cMedQA_pos_ans_only(config):
    # 取出train中所有的question 以及pos answer 作为训练数据
    df_questions, df_answers, df_train, df_dev, df_test = load_and_tokenize_cMedQA(config)
    train_que_ids = list(set(df_train['que_id'].to_list()))
    train_ques = df_questions[df_questions['que_id'].isin(train_que_ids)]
    df_train_join = pd.merge(train_ques, df_answers, left_on="que_id", right_on="ans_que_id")
    df_dev_join = df_dev.merge(df_questions, on='que_id') \
        .merge(df_answers, on="ans_id")[["que_id", "question_tokens", "answer_tokens", "label"]]
    df_test_join = df_test.merge(df_questions, on="que_id") \
        .merge(df_answers, on="ans_id")[["que_id", "question_tokens", "answer_tokens", "label"]]
    train = zip_dataframe(df_train_join, ['que_id', 'question_tokens', 'answer_tokens'])
    dev = zip_dataframe(df_dev_join, ["que_id", "question_tokens", "answer_tokens", "label"])
    test = zip_dataframe(df_test_join, ["que_id", "question_tokens", "answer_tokens", "label"])
    return train, dev, test


def build_dataset_cMedQA_que_only(config):
    # 取出train中所有的question
    df_questions = pd.read_csv(config.question_path) \
        .rename(columns={"content": "question"})[["que_id", "question"]]
    tqdm.pandas(desc="question bar")
    df_questions['question_tokens'] = df_questions['question'].progress_apply(lambda x: tokenizer_and_pad(x, config))
    ques = zip_dataframe(df_questions, ["que_id", "question_tokens"])
    return ques


def build_dataset_CHIPSTS_orgin(config):
    train_path, dev_path, test_path = get_paths_CHIP_STS(config.data_forder)
    df_train = tokenize_df(pd.read_csv(train_path), config.tokenizer, config.pad_size, ['text1', 'text2'])
    df_dev = tokenize_df(pd.read_csv(dev_path), config.tokenizer, config.pad_size, ['text1', 'text2'])
    df_test = tokenize_df(pd.read_csv(test_path), config.tokenizer, config.pad_size, ['text1', 'text2'])
    train = zip_dataframe(df_train, ['text1_tokens', 'text2_tokens', 'label'])
    dev = zip_dataframe(df_dev, ['text1_tokens', 'text2_tokens', 'label'])
    test = zip_dataframe(df_test, ['text1_tokens', 'text2_tokens', 'label'])
    return train, dev, test


def build_dataset_CHIPSTS_pos_only(config):
    train_path, dev_path, test_path = get_paths_CHIP_STS(config.data_forder)
    df_train = tokenize_df(pd.read_csv(train_path), config.tokenizer, config.pad_size, ['text1', 'text2'])
    df_dev = tokenize_df(pd.read_csv(dev_path), config.tokenizer, config.pad_size, ['text1', 'text2'])
    df_test = tokenize_df(pd.read_csv(test_path), config.tokenizer, config.pad_size, ['text1', 'text2'])
    df_train = df_train[df_train['label'] == 1]
    train = zip_dataframe(df_train, ['text1_tokens', 'text2_tokens'])
    dev = zip_dataframe(df_dev, ['text1_tokens', 'text2_tokens', 'label'])
    test = zip_dataframe(df_test, ['text1_tokens', 'text2_tokens', 'label'])
    return train, dev, test

def build_dataset_unsup_cMedQA_dev_CHIPSTS(config):
    questions_path = config.data_forder + "/questions.csv"
    answers_path = config.data_forder + "/answers.csv"
    dev_path = config.data_forder + "/dev_candidates.txt"
    df_question = pd.read_csv(questions_path)
    df_answer = pd.read_csv(answers_path)
    if config.debug_short_set != -1:
        df_question = df_question[:config.debug_short_set]
        df_answer = df_answer[:config.debug_short_set]
    question_df = tokenize_df(df_question, config.tokenizer, config.pad_size, 'content')
    answer_df = tokenize_df(df_answer, config.tokenizer, config.pad_size, 'content')
    train_df = pd.concat([question_df[['content_tokens']], answer_df[['content_tokens']]])
    # train_df = train_df.random_sample(len(train_df))
    dev_df = tokenize_df(pd.read_csv(dev_path), config.tokenizer, config.pad_size, ['text1', 'text2'])
    return zip_dataframe(train_df, ['content_tokens']), \
            zip_dataframe(dev_df, ['text1_tokens', 'text2_tokens', 'label']), \
            zip_dataframe(dev_df, ['text1_tokens', 'text2_tokens', 'label'])



class DatasetIterater(object):
    def __init__(self, batches, config, iter_type):
        self.batch_size = config.batch_size
        self.batches = batches
        self.n_batches = len(batches) // config.batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = config.device
        self.iter_type = iter_type
        self.train_method = config.train_method
        self.data_method = config.data_method

    def _to_tensor(self, datas):
        if self.iter_type == "train":
            if self.data_method == "cMedQA_origin":
                que_tokens = torch.LongTensor([_[0][0] for _ in datas]).to(self.device)
                que_mask = torch.LongTensor([_[0][1] for _ in datas]).to(self.device)
                que_seq_len = torch.LongTensor([_[0][2] for _ in datas]).to(self.device)

                pos_ans_tokens = torch.LongTensor([_[1][0] for _ in datas]).to(self.device)
                pos_ans_mask = torch.LongTensor([_[1][1] for _ in datas]).to(self.device)
                pos_ans_seq_len = torch.LongTensor([_[1][2] for _ in datas]).to(self.device)

                neg_ans_tokens = torch.LongTensor([_[2][0] for _ in datas]).to(self.device)
                neg_ans_mask = torch.LongTensor([_[2][1] for _ in datas]).to(self.device)
                neg_ans_seq_len = torch.LongTensor([_[2][2] for _ in datas]).to(self.device)

                return (que_tokens, que_mask, que_seq_len), \
                       (pos_ans_tokens, pos_ans_mask, pos_ans_seq_len), \
                       (neg_ans_tokens, neg_ans_mask, neg_ans_seq_len)
            elif self.data_method == 'cMedQA_pos_ans_only':
                # 对比学习的训练形式
                que_tokens = torch.LongTensor([_[0][0] for _ in datas]).to(self.device)
                que_mask = torch.LongTensor([_[0][1] for _ in datas]).to(self.device)
                que_seq_len = torch.LongTensor([_[0][2] for _ in datas]).to(self.device)

                ans_tokens = torch.LongTensor([_[1][0] for _ in datas]).to(self.device)
                ans_mask = torch.LongTensor([_[1][1] for _ in datas]).to(self.device)
                ans_seq_len = torch.LongTensor([_[1][2] for _ in datas]).to(self.device)
                return (que_tokens, que_mask, que_seq_len), \
                       (ans_tokens, ans_mask, ans_seq_len)
            elif self.data_method == "cMedQA_que_only":
                que_ids = [_[0] for _ in datas]
                que_tokens = torch.LongTensor([_[1][0] for _ in datas]).to(self.device)
                que_mask = torch.LongTensor([_[1][1] for _ in datas]).to(self.device)
                que_seq_len = torch.LongTensor([_[1][2] for _ in datas]).to(self.device)
                return que_ids, \
                        (que_tokens, que_mask, que_seq_len)
        else:
            que_ids = [_[0] for _ in datas]
            que_tokens = torch.LongTensor([_[1][0] for _ in datas]).to(self.device)
            que_mask = torch.LongTensor([_[1][1] for _ in datas]).to(self.device)
            que_seq_len = torch.LongTensor([_[1][2] for _ in datas]).to(self.device)

            ans_tokens = torch.LongTensor([_[2][0] for _ in datas]).to(self.device)
            ans_mask = torch.LongTensor([_[2][1] for _ in datas]).to(self.device)
            ans_seq_len = torch.LongTensor([_[2][2] for _ in datas]).to(self.device)

            label = torch.LongTensor([_[3] for _ in datas])

            return que_ids,\
                (que_tokens, que_mask, que_seq_len),\
                (ans_tokens, ans_mask, ans_seq_len),\
                label

    def _data_to_tensor(self, datas):
        """
        datas [
            repeated: (list[a],list[b],list[c]) or a
        ]
        若该元素为一维， 则默认为id字段，不转化为tensor直接打进结果返回
        (若该元素为一维dev label， 则在evaluate中单独处理)
        若该元素为多维，处理为tensor ，打进结果返回
        该函数返回结果为对应每一行的每个单元（tuple 或者 int）分别组成tensor
        最后返回的结果形状类似datas 只是按列打成tensor

        isinstance(a, List)
        Out[4]: True
        """
        res = []
        zip_data = zip(*datas)
        for column in zip_data:
            shape = np.array(column).shape
            if len(shape) == 1:
                res.append(column)
            elif len(shape) == 2:
                zip_col = zip(*column)
                res.append([torch.LongTensor(_).to(self.device) for _ in zip_col])
        return res

        # res_map = {} # 存储每一列的tensor ，其key为datas每行数据的 单元（tuple or int）下标
        # for data in datas: # data为一行的元素
        #     # 对每一行进行遍历
        #     for index, unit in enumerate(data): # 对打他中每个单元进行遍历
        #         if index not in res_map:
        #             res_map[index] = []
        #         if isinstance(unit, Tuple): # 如果该单元是tuple，则将tuple内的数据打入tensor ，并且将结果写入res_map
        #             res_map[index].append([torch.LongTensor(_).to(self.device) for _ in unit])
        #         else: # 若该单元不是tuple，在这里默认其为id类型列，不做处理直接打入res_map
        #             res_map[index].append(unit)
        # return [res_map[index] for index in range(len(data))] # 将res_map中的数据按照原有顺序返回

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._data_to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._data_to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, iter_type):
    iter = DatasetIterater(dataset, config, iter_type)
    return iter

def build_dev_generator(dataset, config):
    return DevGenerator(dataset, config)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class DevGenerator(object):
    def __init__(self, dateset, config):
        self.dataset = dateset
        self.config = config

    def new_sample(self, rate):
        que_ids = list(set([d[0] for d in self.dataset]))
        sample_que_id = random.sample(que_ids, int(rate * len(que_ids)))
        sample_data = []
        for data in self.dataset:
            if data[0] in sample_que_id:
                sample_data.append(data)
        iter = DatasetIterater(sample_data, self.config, "dev")
        return iter


if __name__ == '__main__':
    config = Config("")
    # build_dataset_qa(config)


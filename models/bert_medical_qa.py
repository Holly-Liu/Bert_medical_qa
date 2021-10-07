# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.question_path = '/home/users/liuhongli/Models/datasets/cMedQA/questions.csv'
        self.answer_path = '/home/users/liuhongli/Models/datasets/cMedQA/answers.csv'
        self.train_path = '/home/users/liuhongli/Models/datasets/cMedQA/train_candidates.txt'      # 训练集
        self.dev_path = '/home/users/liuhongli/Models/datasets/cMedQA/dev_candidates.txt'          # 验证集
        self.test_path = '/home/users/liuhongli/Models/datasets/cMedQA/test_candidates.txt'         # 测试集
        self.save_path = '/saved_dict/' + dataset + "_" + self.model_name + '.ckpt'        # 模型训练结果
        self.device = "cuda" # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 96                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.margin = 0.5
        self.bert_path = '/home/users/liuhongli/Models/prev_trained_model/bert-base'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.hitn = 1
        self.seed = 1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # out = self.fc(pooled)
        return pooled

# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel



class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.data_forder = '/home/users/liuhongli/Models/datasets/' + dataset
        self.data_method = "cMedQA_all"
        self.train_method = "qa"
        self.eval_method = 'f1'
        self.bert_path = '/home/users/liuhongli/Models/prev_trained_model/mc-bert-torch'
        self.save_path = '/saved_dict/' + dataset + "_" + self.model_name + '.ckpt'        # 模型训练结果
        self.device = "cuda:0" # 设备
        self.debug_short_set = False
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 16                                            # mini-batch大小
        self.pad_size = 96                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.margin = 0.1
        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.hitn = 1
        self.seed = 1
        self.dev_step = 500
        self.dev_rate = 1
        self.test_step = 5000
        self.gama = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.pool = nn.AvgPool2d((config.pad_size-2, 1), stride=(1, 1))
        self.bert.config.hidden_dropout_prob

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        sequence = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # out = self.fc(pooled)

        return self.pool(sequence[:,1:-1,:]).squeeze()

# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from pytorch_pretrained import BertModel



class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        """
        以下配置都不是最终于运行时的配置，运行时会根据输入参数进行同步
        :param dataset: dataset 目录
        """
        self.base_dir = '/home/users/liuhongli/Models/'

        self.model_name = 'bert'
        self.model = 'bert_cons_learning'
        self.eval_method = 'hitn'
        self.log_name = "debuglog"
        self.dev_rate = 1.0
        self.output_size = 100
        self.data_forder = self.base_dir + '/datasets/' + dataset

        self.data_method = 'cMedQA_pos_ans_only'
        self.debug_short_set = False
        self.save_path = '/saved_dict/' + dataset + "_" + self.model_name + '.ckpt'        # 模型训练结果
        self.device = "cuda" # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 96                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.margin = 0.5
        self.bert_path = ''
        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.hitn = 1
        self.seed = 1
        self.dev_step = 500
        self.test_step = 5000
        self.train_method = "simcse"
        self.gama = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.bert.config.hidden_dropout_prob = 0.3
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        # self.pool = nn.AvgPool2d((config.pad_size-2, 1), stride=(1, 1))

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        last_layer, pooled_last_layer = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # out = self.fc(pooled)

        return pooled_last_layer

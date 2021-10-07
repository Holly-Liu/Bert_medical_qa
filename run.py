# coding: UTF-8
import time
import torch
import numpy as np
from tqdm import tqdm

from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_iterator, get_time_dif, build_dev_generator, build_dataset
from my_log import Logger

base_dir = "/home/users/liuhongli/Models/"
prev_trained_model = base_dir + "prev_trained_model/"

parser = argparse.ArgumentParser(description='Chinese Text Classification')

parser.add_argument('--device', type=str, default="cuda:0", help='device eg: cuda:0')

parser.add_argument('--debug_short_set', type=int, default=3000, help='debug short set num')
parser.add_argument('--model', type=str, default="bert_cons_learning", help='choose a model: Bert, ERNIE')
parser.add_argument('--dataset', type=str, default='cMedQA', help='dataset forder')
parser.add_argument('--data_method', type=str, default='cMedQA_pos_ans_only', help='process data method see utils.py')
parser.add_argument('--train_method', type=str, default="cons", help='qa or cons, qa -> question answer cons -> contrast learning')
parser.add_argument('--eval_method', type=str, default="hitn", help='hitn, f1, auc')
parser.add_argument('--bert_path', type=str, default= prev_trained_model + "/mc-bert-torch/", help='bert dir')

parser.add_argument('--log_name', type=str, default="debuglog", help='log file name')
parser.add_argument('--save_path', type=str, default=prev_trained_model + 'cons_learning/', help='model save dir ')

parser.add_argument('--dev_step', type=int, default=10, help='dev per step')
parser.add_argument('--dev_rate', type=float, default=1., help='dev use dev data rate')
parser.add_argument('--test_step', type=int, default=1000, help='test per step')
parser.add_argument('--require_improvement', type=int, default=1000, help='elary stop')
parser.add_argument('--num_epochs', type=int, default=3, help='num epoch')

parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--pad_size', type=int, default=96, help='max seq length')
parser.add_argument('--hidden_size', type=int, default=768, help='hidden size ')
parser.add_argument('--hitn', type=int, default=10, help='n for hitn compute ')
parser.add_argument('--output_size', type=int, default=100, help='output size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='lr')
parser.add_argument('--margin', type=float, default=0.1, help='margin ranking loss margin')
parser.add_argument('--gama', type=float, default=0.1, help='cons loss gama')

parser.add_argument('--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
log = Logger(f"logs/{args.log_name}.log")

def sync_config(args, config):
    dict_args = vars(args)
    log.logger.info("exp params")
    for key, value in dict_args.items():
        log.logger.info(f"{key}: {value}")
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(key, value)
    return config



if __name__ == '__main__':

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(args.dataset)
    log.logger.info("sync confg")
    config = sync_config(args, config)


    np.random.seed(config.seed)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    log.logger.info("Loading data...")
    # train_data, dev_data, test_data = build_dataset_qa(config)
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config, iter_type="train")
    # 由于dev耗时过多， 所以采用对dev进行随机采样的方式，每次eval都采样一部分dev数据
    # dev_generator 函数返回一个对象，可以调用该对象的new_sample 方法获取新的随机dev_iter
    # dev_generator = build_dev_generator(dev_data, config)
    dev_iter = build_iterator(dev_data, config, iter_type='dev')
    test_iter = build_iterator(test_data, config, iter_type="test")
    time_dif = get_time_dif(start_time)
    log.logger.info(f"Time usage:{time_dif}")

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter, log)




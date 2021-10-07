# coding: UTF-8
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam


class BatchContrastLoss(nn.Module):
    def __init__(self, gama):
        super(BatchContrastLoss, self).__init__()
        self.gama = gama

    def forward(self, que_batch, ans_batch):
        que_batch, ans_batch = que_batch.cpu(), ans_batch.cpu()
        total_loss = torch.FloatTensor([0.])
        for i in range(que_batch.shape[0]):
            # 当前分子
            cur_numerator = torch.exp(F.cosine_similarity(que_batch[i], ans_batch[i], dim=0) / self.gama)
            # 当前分母
            cur_denominator = torch.FloatTensor([0.0])
            for j in range(que_batch.shape[0]):
                cur_denominator += torch.exp(F.cosine_similarity(que_batch[i], ans_batch[j], dim=0) / self.gama)
            cur_loss = - torch.log(cur_numerator / cur_denominator)
            total_loss += cur_loss
        return total_loss / que_batch.shape[0]


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, log):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_score = float('-inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    last_test_score = 0
    flag = False  # 记录是否很久没有效果提升
    model.train()
    loss_func = torch.nn.MarginRankingLoss(margin=config.margin) if config.train_method == "qa" else BatchContrastLoss(config.gama)

    for epoch in range(config.num_epochs):
        log.logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        tqdm_dict = {"epoch": epoch}
        for i, train_batch in tqdm(enumerate(train_iter), desc="training", postfix=tqdm_dict):
            model.zero_grad()
            if config.train_method == 'qa':
                que, pos_ans, neg_ans = train_batch
                que_outputs = model(que)
                pos_ans_outputs = model(pos_ans)
                neg_ans_outputs = model(neg_ans)
                pos_sim = F.cosine_similarity(que_outputs, pos_ans_outputs)
                neg_sim = F.cosine_similarity(que_outputs, neg_ans_outputs)
                loss = loss_func(pos_sim, neg_sim, torch.ones_like(pos_sim))
            elif config.train_method == "cons":
                que_ids, que, ans = train_batch
                que_outputs = model(que)
                ans_outputs = model(ans)
                loss = loss_func(que_outputs, ans_outputs)
            elif config.train_method == "simcse":
                que, ans = train_batch
                que_out1 = model(que)
                que_out2 = model(que)
                ans_out1 = model(ans)
                ans_out2 = model(ans)
                loss_que = loss_func(que_out1, que_out2)
                loss_ans = loss_func(ans_out1, ans_out2)
                loss = loss_que + loss_ans
            elif config.train_method == "sim_cls":
                text1, text2, label = train_batch
                text1_out, text2_out = model(text1), model(text2)
                sim = F.cosine_similarity(text1_out, text2_out)
                loss = F.cross_entropy(sim, label)
            elif config.train_method == "unsup_simcse":
                model.bert.config.hidden_dropout_prob = 0.3
                text = train_batch[0]
                out1 = model(text)
                out2 = model(text)
                loss = loss_func(out1, out2)
            loss.backward()
            optimizer.step()
            total_batch += 1
            if total_batch % 10 == 0 and total_batch % config.dev_step != 0:
                log.logger.info(f"epoch: {epoch} batch[{i}/{train_iter.n_batches}], train loss: {loss.item()}")
            if total_batch % config.dev_step == 0:
                # 每多少轮输出在训练集和验证集上的效果, dev_generator.new_sample() 返回一个新的sample
                dev_acc, dev_score = evaluate(config, model, dev_iter, log)
                if dev_score > dev_best_score:
                    dev_best_score = dev_score
                    if not os.path.exists(config.save_path):
                        os.mkdir(config.save_path)
                    torch.save(model.bert.state_dict(), config.save_path + "/pytorch_model.bin")
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = f'{improve} Iter: {total_batch},  Train Loss: {loss.item()},  Val f1: {dev_acc}, Val auc: {dev_score}, Time: {time_dif} {improve}'
                log.logger.info(msg)
                model.train()
            if total_batch % config.test_step == 0:
                log.logger.info("testing ")
                test_acc, test_score = evaluate(config, model, test_iter, log)
                last_test_score = max(last_test_score, test_score)
                msg = f'Iter: {total_batch}, test f1: {test_acc}, test auc: {test_score}, Time: {time_dif} '
                log.logger.info(msg)
            # if total_batch - last_improve > config.require_improvement:
            #     # 如果1000个batch在验证集上效果没提升，验证在测试集上的效果
            #     log.logger.info("test set eval")
            #     # test_acc, test_score = evaluate(config, model, test_iter, log)
            #     # if test_score <= last_test_score:
            #     log.logger.info("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
                # else:
                #     last_improve = total_batch
                #     last_test_score = test_score
        if flag:
            break
        f1, auc = evaluate(config, model, dev_iter, log)
        log.logger.info(f"epoch: {epoch}, test_f1: {f1}, test auc: {auc}")
    f1, auc = evaluate(config, model, dev_iter, log)
    log.logger.info(f"test_f1: {f1}, test auc: {auc}")


def evaluate(config, model, data_iter, log):
    model.eval()
    if config.eval_method == "hitn":
        return eval_hitn(config.hitn, model, data_iter, log)
    if config.eval_method == "auc":
        return eval_auc(model, data_iter, log)


def eval_hitn(n, model, data_iter, log):
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    que_id_all = np.array([], dtype=int)
    log.logger.info("model evaluating")
    with torch.no_grad():
        for que_ids, que, ans, labels in data_iter:
            que_output = model(que)
            ans_output = model(ans)
            sim_score = F.cosine_similarity(que_output, ans_output)
            labels = np.array(labels) # labels 没有放进tensor
            predic = sim_score.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            que_id_all = np.append(que_id_all, que_ids)
    acc, dev_score = hitn_acc(que_id_all, labels_all, predict_all, log, n=n)
    return acc, dev_score

def hitn_acc(que_id_all, labels_all, predict_all, log, n=1):
    score_map = {}
    score_all = []
    log.logger.info("computing hitn")
    # 计算dev_score 计算方式为所有label=1的样例排序后的index倒数相加
    dev_score = 0
    for i in tqdm(range(len(que_id_all)), desc="evaluating"):

        que_id, predict, label = que_id_all[i], predict_all[i], labels_all[i]
        if que_id not in score_map:
            score_map[que_id] = []
        score_map[que_id].append((predict, label))
    for key in score_map.keys():
        score_map[key].sort(key=lambda x: -x[0])
        # 计算当前que_id 有多少个label = 1 的样例
        count_origin = sum([l for p, l in score_map[key]])
        # 计算dev_score
        dev_score += sum([1/(i+1) for i, (p, l) in enumerate(score_map[key]) if l == 1])
        score_map[key] = score_map[key][:n]
        # 计算排名前n的 样例中有多少label = 1
        count_hit = sum([l for p, l in score_map[key]])
        score = count_hit / (min(n, count_origin) + 0.0001)
        score_all.append(score)
    return sum(score_all) / len(score_all), dev_score


def eval_auc(model, data_iter, log):
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for text1, text2, label in data_iter:
            sim = F.cosine_similarity(model(text1), model(text2))
            predict_all = np.append(predict_all, sim.cpu().numpy())
            labels_all = np.append(labels_all, np.array(label))
    auc = metrics.roc_auc_score(labels_all, predict_all)
    lp = predict_all.tolist()
    threshold = sorted(lp)[int(len(predict_all) // 2)]
    predict_all[predict_all > threshold] = 1
    predict_all[predict_all <= threshold] = 0
    predict_all = predict_all.astype(np.int64)
    f1 = metrics.f1_score(labels_all, predict_all)
    return f1, auc








import time
import torch
import numpy as np
from importlib import import_module   # 动态导入
import argparse
import utils
import train   


parser = argparse.ArgumentParser(description='Bert-Text-Classsification')
parser.add_argument('--model', type=str, default='Bert', help = 'choose a model Bert, BertCNN, BertRNN')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    np.random.seed(1)      
    torch.manual_seed(1)       
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()

    print('加载数据集')
    train_data, dev_data, test_data = utils.bulid_dataset(config)
    train_iter = utils.bulid_iterator(train_data, config)
    dev_iter = utils.bulid_iterator(dev_data, config)
    test_iter = utils.bulid_iterator(test_data, config)

    # # 输出调试
    # for i, (trains, labels) in enumerate(train_iter):
    #     print(i, labels)

    time_dif = utils.get_time_dif(start_time)
    print("模型开始之前，准备数据时间：", time_dif)

    #模型
    model = x.Model(config).to(config.device)
    # 模型训练，评估与测试
    # train.train(config, model, train_iter, dev_iter, test_iter)
    train.test(config, model, test_iter)

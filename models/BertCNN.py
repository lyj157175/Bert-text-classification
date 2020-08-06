import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        # 模型名称
        self.model_name="BruceBertCNN"
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        #dataset
        self.datasetpkl = dataset + '/data/dataset.pkl'
        # 类别名单
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型保存路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvement= 1000
        # 类别数量
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 128
        # 序列长度
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # 预训练位置
        self.bert_path = './bert_pretrain'
        # bert的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained((self.bert_path))
        # Bert的隐藏层数量
        self.hidden_size = 768
        # 卷积核尺寸
        self.filter_sizes = (2,3,4)
        # 卷积核数量
        self.num_filters = 256
        # droptout
        self.dropout = 0.5

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        #调用Conv2d()做卷积
        # conv2d:
        # in_channels: 文本为1
        # out_channels: 256  输出多少个卷积核
        # kernel_size(tuple)=(k, hiiden_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.hidden_size)) for k in config.filter_sizes]
        )

        self.droptout = nn.Dropout(config.dropout)
        #[卷积核数量*卷积种类，输出类别]
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):  # x:[128, 1, 32, 768]    (seq_len + 2p - f)/s + 1
        x = conv(x)    # x:[128, 256, 31, 1]
        x = F.relu(x)
        x = x.squeeze(3)  #x: [128, 256, 31]
        size = x.size(2)
        x = F.max_pool1d(x, size)  # [128, 256, 1]
        x = x.squeeze(2)
        return x   # [128, 256]


    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0] #对应输入的句子 shape[128,32]
        mask = x[2] #对padding部分进行mask shape[128,32]


        # False返回小bert的最后一层，True返回全部的12层
        # encoder_out：[batch_size, seq_len, bert_dim]
        # pooled: [batch_size, bert_dim]
        # [128, 32, 768] [128,768]
        encoder_out, pooled = self.bert(context, attention_mask = mask, output_all_encoded_layers = False)
        out = encoder_out.unsqueeze(1)  # [batch_size, seq_len, hidden_size]-> [128， 1， 32， 768] [b，通道数，句子高度，句子宽度]（conv2d需要输入四维数据）
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  #[128, 256*3]
        out = self.droptout(out)
        out = self.fc(out)  # [128, 10]
        return out




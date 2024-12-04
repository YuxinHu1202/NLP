import torch
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# def get_best_device():
#     if(torch.cuda.is_available):
#         best_device=None
#         best_memory=0
#         for i in range(torch.cuda.device_count()):
#             device=torch.device(f"cuda:{i}")
#             available=get_gpu_memory()[i]
#             if(available)
# 初始化模型和分词器
model_name = 'bert-base-uncased'
model=BertModel.from_pretrained(model_name)#加载预训练的BERT模型（bert-base-uncased）。
tokenizer=BertTokenizer.from_pretrained("./bert-base-uncased")#加载与BERT模型配套的分词器。这里指定的路径是本地路径./bert-base-uncased
#data=pd.read_csv("data/adult.tsv",sep='\t')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=get_best_device()
model.to(device)

# 定义一个线性分类器
# 假设任务是二分类任务s
num_labels = 2  # number of classed
#是一个全连接层，表示线性分类器。该层会将输入的特征映射到输出的类别数（即二分类任务中的2个标签）。
classifier = nn.Linear(model.config.hidden_size, num_labels).to(device)

#定义一个三层感知机（MLP）
dk=300 #定义了MLP最后一层的输出维度为300。即，经过MLP处理后的输出将是一个300维的向量。
mlp=nn.Sequential(#这将构建一个顺序容器，其内部的层将按顺序执行。
    nn.Linear(model.config.hidden_size,model.config.hidden_size),#这是一个全连接层，输入的维度和输出的维度都等于model.config.hidden_size，也就是BERT模型的输出维度。
    nn.ReLU(),#这是一个ReLU激活函数，通常用于增加非线性性
    nn.Linear(model.config.hidden_size,dk),#这是另一个全连接层，将输入的维度从model.config.hidden_size映射到d_k（即300维）。
).to(device)

# 初始化优化器
# Change optimizer to AdamW with weight decay 0.01
#初始化了一个AdamW优化器。AdamW是Adam优化器的一个变种，支持权重衰减
#这里的model.parameters()返回BERT模型的所有可学习参数，classifier.parameters()返回分类器（classifier）的所有可学习参数。
#lr=2e-5：设置学习率为2e-5,weight_decay=0.01：设置权重衰减（L2正则化）的系数为0.01
optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-5, weight_decay=0.01)

#epsilon 是一个常见的超参数，通常用于机器学习和深度学习任务中，尤其是在对抗训练、优化算法、数值计算中的步长等情境下。
#对抗训练：在对抗样本生成中，epsilon 表示扰动的幅度，控制添加到输入数据中的扰动大小。较小的 epsilon 值会产生微小的扰动，而较大的 epsilon 值会产生更强的扰动。通常用于生成对抗攻击（如FGSM攻击）时。
# epsilon = [0.0001,0.001,0.005,0.02]
epsilon = 0.001


# 加载SST-2数据集
train_file = 'data/SST-2/train.tsv'
dev_file = 'data/SST-2/dev.tsv'
test_file = 'data/SST-2/test_nolabel.tsv'

train_data = pd.read_csv(train_file, sep='\t')
dev_data = pd.read_csv(dev_file, sep='\t')
test_data = pd.read_csv(test_file, sep='\t')

class SST2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']  # 使用SST-2中的文本列
        label = self.data.iloc[idx]['label']  # 使用标签列
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 创建训练集和验证集数据加载器
train_dataset = SST2Dataset(train_data, tokenizer)
dev_dataset = SST2Dataset(dev_data, tokenizer)


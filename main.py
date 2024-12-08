import torch
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 初始化模型和分词器
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)  # 加载预训练的BERT模型（bert-base-uncased）。
tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")  # 加载与BERT模型配套的分词器。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义一个线性分类器
num_labels = 2  # number of classes
classifier = nn.Linear(model.config.hidden_size, num_labels).to(device)

# 定义一个三层感知机（MLP）
dk = 300  # 最后一层的输出维度
mlp = nn.Sequential(
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
    nn.ReLU(),
    nn.Linear(model.config.hidden_size, dk),
).to(device)

# 初始化优化器
optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-5, weight_decay=0.01)

epsilon = 0.001  # 对抗训练的扰动幅度

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
        text = self.data.iloc[idx]['text_a']
        label = self.data.iloc[idx]['label']
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

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

accuracy_list = []  # 用于记录每个epoch的准确率
loss_list = []  # 用于记录每个epoch的损失

epochs = 1

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))

        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)
        self.register_buffer("negatives_mask", negatives_mask.float())
    
    def forward(self, emb_i, emb_j):
        z_i = nn.functional.normalize(emb_i, dim=1)
        z_j = nn.functional.normalize(emb_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


for epoch in range(epochs):
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    model.train()
    classifier.train()
    
    for i, batch in enumerate(train_loader):
        if i >= 2:  # 仅使用前两个批次进行测试
            break
        
        inputs = {name: tensor.to(model.device) for name, tensor in batch.items() if name in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(model.device)
        
        # 获取BERT的输出
        outputs = model(**inputs)[0][:, 0, :]
        
        # 计算原始的分类器损失
        logits = classifier(outputs)
        original_loss = nn.functional.cross_entropy(logits, labels)
        
        # 计算损失函数关于输入的梯度
        model.embeddings.word_embeddings.weight.requires_grad = True
        original_loss.backward(retain_graph=True)
        
        # 应用FGSM
        normalized_gradient = torch.nn.functional.normalize(model.embeddings.word_embeddings.weight.grad.data, p=2)
        perturbed_embeddings = model.embeddings.word_embeddings.weight - epsilon * normalized_gradient
        perturbed_embeddings.detach_()  # 使用detach()来进行梯度分离，确保对抗样本不影响模型的其他部分
        
        model.embeddings.word_embeddings.weight.data = perturbed_embeddings
        perturbed_outputs = model(**inputs)[0][:, 0, :]
        
        # 计算扰动后的分类器损失
        perturbed_logits = classifier(perturbed_outputs)
        perturbed_loss = nn.functional.cross_entropy(perturbed_logits, labels)

        # 计算MLP损失
        mlp_outputs = mlp(outputs)
        mlp_perturbed_outputs = mlp(perturbed_outputs)

        # 对比损失
        t = 0.05  # 温度系数
        loss_func = ContrastiveLoss(batch_size=batch_size, temperature=t)
        mlp_similarity_loss = loss_func(mlp_outputs, mlp_perturbed_outputs)
        
        # 计算总损失
        Lambda = 0.5  # 权重系数
        total_batch_loss = (1 - Lambda) * (original_loss + perturbed_loss) / 2 + mlp_similarity_loss * Lambda
        total_loss += total_batch_loss.item()
        
        # 反向传播和优化
        total_batch_loss.backward()
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 清空当前梯度
        
        # 计算准确率
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_examples += labels.size(0)
        
        print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {total_batch_loss.item()}, Accuracy: {correct / labels.size(0)}")
    
    torch.save(model.state_dict(), f"result2/model_epoch_{epoch}.pt")
    
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss}")

    avg_accuracy = total_correct / total_examples
    accuracy_list.append(avg_accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Avg Accuracy: {avg_accuracy}")

# 保存损失图表
plt.figure()
plt.plot(range(epochs), loss_list)
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('result2/loss.png')  # Save the figure
plt.close()

# 保存准确率图表
plt.figure()
plt.plot(range(epochs), accuracy_list)
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('result2/accuracy.png')  # Save the figure
plt.close()

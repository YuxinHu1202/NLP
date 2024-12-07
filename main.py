import torch
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
        text = self.data.iloc[idx]['text_a']  # 使用SST-2中的文本列
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

# 打印数据集的列名，检查列是否为 'sentence' 和 'label'
print(train_data.columns)


batch_size=8
#shuffle=True 表示数据将在每个epoch开始时被打乱，这样有助于减少训练过程中的偏差。
#drop_last=True 意味着如果数据集的大小不能被 batch_size 整除，最后一个批次会被丢弃。
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
dev_loader=DataLoader(dev_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

accuracy_list=[]#accuracy_list：用于记录每个epoch的准确率
loss_list = []#用于记录每个epoch的损失（通常是对比损失）。

epochs=1

class ContrastiveLoss(nn.Module):
    #temperature：温度超参数，它控制了对比损失的敏感度。温度较高时，模型的区分度较低，较低时，模型的区分度较高
    def __init__(self, batch_size, temperature=0.05):
        super().__init__()
        self.batch_size=batch_size

        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        #negatives_mask：这个掩码矩阵是一个对角线为 0、其它位置为 1 的矩阵，大小为 (batch_size * 2, batch_size * 2)。它用于排除正样本之间的相似度计算。
        #换句话说，掩码矩阵将正样本的相似度值（主对角线）排除在计算之外，使其只计算负样本对的相似度。
        # self.register_buffer("negatives_mask",(
        #     ~torch.eye(batch_size*2,batch_size*2,dtype=bool).to(device)).float())
        # 创建 negatives_mask
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)

        # 注册 negatives_mask
        self.register_buffer("negatives_mask", negatives_mask.float())  # 将布尔型转换为 float 类型
    
    #这两个输入分别表示原始文本的嵌入（emb_i）和经过扰动后的文本嵌入（emb_j）。这两个嵌入通过对比学习来计算相似度。
    def forward(self,emb_i,emb_j):#这两个输入分别表示原始文本的嵌入（emb_i）和经过扰动后的文本嵌入（emb_j）。这两个嵌入通过对比学习来计算相似度。
        #nn.functional.normalize()：这行代码对 emb_i 和 emb_j 进行 L2 标准化，使它们的每个向量长度都为 1。
        z_i=nn.functional.normalize(emb_i,dim=1)
        z_j=nn.functional.normalize(emb_j,dim=1)
        
        representations=torch.cat([z_i,z_j],dim=0)#将 emb_i 和 emb_j 按行拼接（torch.cat），得到一个新的张量，表示所有正负样本的嵌入。
        #计算所有嵌入之间的余弦相似度。使用 cosine_similarity 来衡量每对嵌入之间的相似度。相似度越高，表示两个样本越相似。
        similarity_matrix=nn.functional.cosine_similarity(representations.unsqueeze(1),representations.unsqueeze(0),dim=2)
        
        #sim_ij 和 sim_ji：提取正样本的相似度。sim_ij 是从 emb_i 到 emb_j 的相似度，而 sim_ji 是从 emb_j 到 emb_i 的相似度。
        sim_ij=torch.diag(similarity_matrix,self.batch_size)
        sim_ji=torch.diag(similarity_matrix,-self.batch_size)
        
        positives=torch.cat([sim_ij,sim_ji],dim=0)
        
        nominator=torch.exp(positives/self.temperature)#计算的是正样本的指数相似度。
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)#计算所有负样本的指数相似度，并通过 negatives_mask 排除正样本对之间的相似度。

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))#这是每个样本的对比损失，使用了 Softmax 方法。损失是通过对正样本与所有负样本之间的相似度进行对比计算得到的。
        loss = torch.sum(loss_partial) / (2 * self.batch_size)#最终的对比损失是所有样本的对比损失的平均值。由于使用了标准化嵌入和指数函数，所以该损失能够有效地拉近相似样本之间的距离，并推远不同样本之间的距离。
        return loss



for epoch in range(epochs):
    total_loss=0
    total_correct=0
    total_examples=0
    
    #开始训练模式
    model.train()
    classifier.train()
    
    for i,batch in enumerate(train_loader):
        if i>=2:
            break
        
        inputs = {name: tensor.to(model.device) for name, tensor in batch.items() if name in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(model.device)
        
        # 计算模型的输出
        # 注意 这里的outputs指的就是[CLS],是 Classification（分类）标记的缩写，模型将通过这个标记来进行分类任务。的输出
        outputs=model(**inputs)[0][:,0,:]
        
        # 计算原始的分类器损失
        logits = classifier(outputs)
        #nn.functional.cross_entropy(logits, labels): 使用交叉熵损失函数计算模型输出（logits）与真实标签（labels）之间的差异
        original_loss = nn.functional.cross_entropy(logits, labels)
        
        # 计算损失函数关于输入的梯度
        #这一行确保嵌入层的权重（即词嵌入矩阵）是可训练的，并且在反向传播时会计算梯度。
        model.embeddings.word_embeddings.weight.requires_grad = True
        original_loss.backward(retain_graph=True)#这行代码进行反向传播计算梯度，梯度会存储在模型的各个参数中（如嵌入层的权重）。retain_graph=True 参数的作用是保留计算图，使得后续的梯度计算可以使用。
        
        # 应用FGSM
        #model.embeddings.word_embeddings.weight.grad.data: 这是嵌入层的权重的梯度，表示损失函数对嵌入权重的导数。
        #torch.nn.functional.normalize(..., p=2): 这行代码使用 L2 正规化（即计算梯度的单位向量），通过对梯度进行标准化，使得梯度的范数为 1。这是对抗训练中常用的操作，以保证梯度方向的准确性，避免过大的梯度扰动。
        normalized_gradient = torch.nn.functional.normalize(model.embeddings.word_embeddings.weight.grad.data, p=2)
        #这一行代码应用 FGSM。FGSM 是一种对抗攻击方法，通过在原始输入（这里是词嵌入）上添加扰动来生成对抗样本。
        #具体地，通过梯度的符号方向对词嵌入进行扰动，epsilon 是扰动的幅度（即扰动的大小），这决定了生成的对抗样本的“强度”。normalized_gradient 是已经标准化的梯度，epsilon 控制扰动的大小。
        perturbed_embeddings = model.embeddings.word_embeddings.weight - epsilon * normalized_gradient
        
        # 将对抗样本输入模型
        model.embeddings.word_embeddings.weight.data = perturbed_embeddings
        perturbed_outputs = model(**inputs)[0][:, 0, :]
        
        # 计算扰动后的分类器损失
        #perturbed_logits = classifier(perturbed_outputs)
        #perturbed_loss = nn.functional.cross_entropy(perturbed_logits, labels)

        # 计算MLP损失
        mlp_outputs = mlp(outputs)
        #mlp_perturbed_outputs = mlp(perturbed_outputs)

        # 温度系数
        # t=[0.05,0.06,0.07,0.08,0.09,0.10]
        #这段代码的作用是创建一个 ContrastiveLoss（对比损失）对象，并使用它计算 MLP 输出的相似度损失
        t = 0.05#较小的温度（例如 0.05）会使得相似样本之间的相似度增加，而不同样本之间的相似度降低。
        loss_func = ContrastiveLoss(batch_size=batch_size, temperature=t)
        #mlp_similarity_loss = loss_func(mlp_outputs, mlp_perturbed_outputs)
       
        # 计算总损失
        # Lambda = [0.1,0.2,0.3,0.4,0.5]
        Lambda = 0.5#Lambda 是一个超参数，控制 对比损失 和 原始损失/扰动损失 之间的加权系数。它的值为 0.5，意味着 原始损失/扰动损失 和 对比损失 在总损失中的贡献是相等的。
        #total_batch_loss = (1 - Lambda) * (original_loss + perturbed_loss) / 2 + mlp_similarity_loss * Lambda#这一部分计算的是 原始损失（original_loss）和 扰动损失（perturbed_loss）的加权平均。
        #mlp_similarity_loss * Lambda这一部分是 对比损失（mlp_similarity_loss）的加权部分。
        #total_loss += total_batch_loss.item()#total_loss 是用来记录每个训练批次的总损失。total_batch_loss.item() 会返回当前批次的总损失值（以标量的形式）。
        total_batch_loss=original_loss
        total_loss += total_batch_loss.item()
        
        # 反向传播和优化
        #total_batch_loss.backward() # 反向传播，计算梯度
        original_loss.backward()
        optimizer.step() ## 更新模型参数
        optimizer.zero_grad()  # 清空当前梯度
        
        # 计算准确率
        _, predicted = torch.max(logits, 1)## 获取预测的类别标签
        correct = (predicted == labels).sum().item() ## 计算预测正确的样本数量
        total_correct += correct  # 累加正确的样本数量
        total_examples += labels.size(0) # 累加总的样本数量
        
                # 打印每个批次的信息
        print(
            f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {total_batch_loss.item()}, Accuracy: {correct / labels.size(0)}")
        
    # 每个epoch结束后，保存模型
    torch.save(model.state_dict(), f"result1/model_epoch_{epoch}.pt")
    
    # 计算并打印平均损失
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss}")

    # 计算并打印准确率
    avg_accuracy = total_correct / total_examples
    accuracy_list.append(avg_accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Avg Accuracy: {avg_accuracy}")

# loss
plt.figure()
plt.plot(range(epochs), accuracy_list)
plt.title('loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('result1/loss.png')  # Save the figure
plt.close()

# Accuracy
plt.figure()
plt.plot(range(epochs), accuracy_list)
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('result1/accuracy.png')  # Save the figure
plt.close()



        

    

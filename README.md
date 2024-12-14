# AGN--通过自适应门控合并统计特征以改善文本分类

## 数据准备

### 数据集
文章中提到的数据集有六个，分别为`Subj`,`SST-1`,`SST-2`,`TREC`,`AG's News`,`Yelp P./F/`,我在复现实验中使用到了如下两个。
| 数据集 | URL |
|-------|-----|
| Yelp  | [http://www.cs.cornell.edu/people/pabo/movie-review-data/](http://www.cs.cornell.edu/people/pabo/movie-review-data/) |
| SST-1 | [http://nlp.stanford.edu/sentiment/](http://nlp.stanford.edu/sentiment/) |

下载完数据集之后，需要将txt格式的数据转换为 JSON 格式，编写了`changeto_json1`和`changeto_json2`两个函数进行格式转换，部分转换结果如下所示：
其中每一行都是一个 JSON 对象，包含两个字段：`text` 和 `label`。

```json
{"label": 1, "text": "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before ."}
{"label": 1, "text": "for starters , it was created by alan moore ( and eddie campbell ) , who brought the medium to a whole new level in the mid '80s with a 12-part series called the watchmen ."}
{"label": 1, "text": "to say moore and campbell thoroughly researched the subject of jack the ripper would be like saying michael jackson is starting to look a little odd ."}
{"label": 0, "text": "plot : two teen couples go to a church party , drink and then drive ."}
{"label": 0, "text": "they get into an accident ."}
{"label": 0, "text": "one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares ."}
```

### 预训练语言模型
我使用了预训练的 **Uncased-Bert-Base** 模型。

## 环境配置

一些特殊的环境配置版本包括如下所示，需要注意tensorflow的版本不能过高！！

```bash
$ pip list | egrep "tensorflow|Keras|langml"
Keras                            2.3.1
langml                           0.1.0
tensorflow                       1.15.0
```

接下来，使用requirements安装其他依赖即可，如再有包缺失报错再安装即可。

```bash
$ python -m pip install -r requirements.txt
```

## 训练与评估

首先在config.json这个配置文件中设置我们的数据路径和以及一下训练超参数，如下所示：

```json
config.json

{
  "max_len": 80,
  "ae_epochs": 100,
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.00003,
  "pretrained_model_dir": "uncased_L-12_H-768_A-12",
  "train_path": "SST-2/train.jsonl",
  "dev_path": "SST-2/test.jsonl",
  "save_dir": "save",
  "epsilon": 0.05,
  "dropout": 0.3,
  "fgm_epsilon": 0.3,
  "iterations": 1,
  "verbose": 1
}
```

超参数的对应解释如下表格所示：

| 参数                   | 描述                              |
|------------------------|-----------------------------------|
| `max_len`              | 输入序列的最大长度                 |
| `ae_epochs`            | 训练自编码器的轮数                |
| `epochs`               | 训练分类器的轮数                  |
| `batch_size`           | 批量大小                           |
| `learning_rate`        | 学习率                             |
| `pretrained_model_dir` | 预训练语言模型的文件目录         |
| `save_dir`             | 保存模型的目录                    |
| `train_path`           | 训练集数据路径                    |
| `dev_path`             | 验证集数据路径          |
| `epsilon`              | 门控的 epsilon 大小               |
| `apply_fgm`            | 是否应用 FGM 攻击，默认为 `true`  |
| `fgm_epsilon`          | FGM 攻击的 epsilon，默认为 0.2    |

在`Windows`系统下：

设置`set TF_KERAS=1`以使用`AdamW`优化器。

设置`set CUDA_VISIBLE_DEVICES=0`以使用`GPU`资源。

使用如下指令开始训练并评估模型：
```bash
 python main.py config.json
```

训练完成后，模型将保存在config.json中指定的 `save_dir` 文件夹中，文件夹的结构如下图所示：
![image](https://github.com/user-attachments/assets/a54ae513-533f-4237-9ae2-a5520fb9b2b4)

训练和评估的可能部分输出结果如下：
```bash
(text2) C:\Users\Lenovo\Desktop\大三上\自然语言处理\跑模型\AGN-main>python main.py config.json
config:
{'ae_epochs': 100,
 'batch_size': 32,
 'dev_path': 'SST2/test.jsonl',
 'dropout': 0.5,
 'epochs': 30,
 'epsilon': 0.05,
 'fgm_epsilon': 0.3,
 'iterations': 1,
 'learning_rate': 3e-05,
 'max_len': 80,
 'pretrained_model_dir': 'uncased_L-12_H-768_A-12',
 'save_dir': 'save2',
 'steps_per_epoch': 32,
 'train_path': 'SST2/train.jsonl',
 'verbose': 1}
successful!
load data...
batch alignment...
previous data size: 1000
alignment data size: 992
set tcol....
token size: 5148
done to set tcol...
train vae...
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
train size: 704
dev size: 288
Train on 704 samples, validate on 288 samples
Epoch 1/100
704/704 - 1s - loss: 0.7593 - val_loss: 0.7106
Epoch 2/100
704/704 - 0s - loss: 0.6667 - val_loss: 0.6114
Epoch 3/100
704/704 - 0s - loss: 0.5747 - val_loss: 0.5393
.
.
.
batch alignment...
previous data size: 199
alignment data size: 192
train vae...
build generator

Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Skip Embedding-Mapping
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
Input-Token (InputLayer)        [(None, None)]       0
__________________________________________________________________________________________________
Input-Segment (InputLayer)      [(None, None)]       0
__________________________________________________________________________________________________
Embedding-Token (TokenEmbedding [(None, None, 768),  23440896    Input-Token[0][0]
__________________________________________________________________________________________________
Embedding-Segment (Embedding)   (None, None, 768)    1536        Input-Segment[0][0]
__________________________________________________________________________________________________
Embedding-Token-Segment (Add)   (None, None, 768)    0           Embedding-Token[0][0]
                                                                 Embedding-Segment[0][0]
__________________________________________________________________________________________________
Embedding-Position (AbsolutePos (None, None, 768)    393216      Embedding-Token-Segment[0][0]
__________________________________________________________________________________________________
Embedding-Norm (LayerNorm)      (None, None, 768)    1536        Embedding-Position[0][0]
__________________________________________________________________________________________________
Embedding-Dropout (Dropout)     (None, None, 768)    0           Embedding-Norm[0][0]
__________________________________________________________________________________________________
Transformer-1-MultiHeadSelfAtte (None, None, 768)    2362368     Embedding-Dropout[0][0]
__________________________________________________________________________________________________
Transformer-1-MultiHeadSelfAtte (None, None, 768)    0           Transformer-1-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-1-MultiHeadSelfAtte (None, None, 768)    0           Embedding-Dropout[0][0]
                                                                 Transformer-1-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-1-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-1-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-1-FeedForward (Feed (None, None, 768)    4722432     Transformer-1-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-1-FeedForward-Dropo (None, None, 768)    0           Transformer-1-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-1-FeedForward-Add ( (None, None, 768)    0           Transformer-1-MultiHeadSelfAttent
                                                                 Transformer-1-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-1-FeedForward-Norm  (None, None, 768)    1536        Transformer-1-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-2-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-1-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-2-MultiHeadSelfAtte (None, None, 768)    0           Transformer-2-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-2-MultiHeadSelfAtte (None, None, 768)    0           Transformer-1-FeedForward-Norm[0]
                                                                 Transformer-2-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-2-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-2-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-2-FeedForward (Feed (None, None, 768)    4722432     Transformer-2-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-2-FeedForward-Dropo (None, None, 768)    0           Transformer-2-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-2-FeedForward-Add ( (None, None, 768)    0           Transformer-2-MultiHeadSelfAttent
                                                                 Transformer-2-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-2-FeedForward-Norm  (None, None, 768)    1536        Transformer-2-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-3-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-2-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-3-MultiHeadSelfAtte (None, None, 768)    0           Transformer-3-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-3-MultiHeadSelfAtte (None, None, 768)    0           Transformer-2-FeedForward-Norm[0]
                                                                 Transformer-3-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-3-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-3-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-3-FeedForward (Feed (None, None, 768)    4722432     Transformer-3-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-3-FeedForward-Dropo (None, None, 768)    0           Transformer-3-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-3-FeedForward-Add ( (None, None, 768)    0           Transformer-3-MultiHeadSelfAttent
                                                                 Transformer-3-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-3-FeedForward-Norm  (None, None, 768)    1536        Transformer-3-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-4-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-3-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-4-MultiHeadSelfAtte (None, None, 768)    0           Transformer-4-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-4-MultiHeadSelfAtte (None, None, 768)    0           Transformer-3-FeedForward-Norm[0]
                                                                 Transformer-4-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-4-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-4-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-4-FeedForward (Feed (None, None, 768)    4722432     Transformer-4-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-4-FeedForward-Dropo (None, None, 768)    0           Transformer-4-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-4-FeedForward-Add ( (None, None, 768)    0           Transformer-4-MultiHeadSelfAttent
                                                                 Transformer-4-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-4-FeedForward-Norm  (None, None, 768)    1536        Transformer-4-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-5-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-4-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-5-MultiHeadSelfAtte (None, None, 768)    0           Transformer-5-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-5-MultiHeadSelfAtte (None, None, 768)    0           Transformer-4-FeedForward-Norm[0]
                                                                 Transformer-5-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-5-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-5-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-5-FeedForward (Feed (None, None, 768)    4722432     Transformer-5-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-5-FeedForward-Dropo (None, None, 768)    0           Transformer-5-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-5-FeedForward-Add ( (None, None, 768)    0           Transformer-5-MultiHeadSelfAttent
                                                                 Transformer-5-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-5-FeedForward-Norm  (None, None, 768)    1536        Transformer-5-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-6-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-5-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-6-MultiHeadSelfAtte (None, None, 768)    0           Transformer-6-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-6-MultiHeadSelfAtte (None, None, 768)    0           Transformer-5-FeedForward-Norm[0]
                                                                 Transformer-6-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-6-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-6-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-6-FeedForward (Feed (None, None, 768)    4722432     Transformer-6-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-6-FeedForward-Dropo (None, None, 768)    0           Transformer-6-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-6-FeedForward-Add ( (None, None, 768)    0           Transformer-6-MultiHeadSelfAttent
                                                                 Transformer-6-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-6-FeedForward-Norm  (None, None, 768)    1536        Transformer-6-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-7-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-6-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-7-MultiHeadSelfAtte (None, None, 768)    0           Transformer-7-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-7-MultiHeadSelfAtte (None, None, 768)    0           Transformer-6-FeedForward-Norm[0]
                                                                 Transformer-7-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-7-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-7-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-7-FeedForward (Feed (None, None, 768)    4722432     Transformer-7-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-7-FeedForward-Dropo (None, None, 768)    0           Transformer-7-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-7-FeedForward-Add ( (None, None, 768)    0           Transformer-7-MultiHeadSelfAttent
                                                                 Transformer-7-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-7-FeedForward-Norm  (None, None, 768)    1536        Transformer-7-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-8-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-7-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-8-MultiHeadSelfAtte (None, None, 768)    0           Transformer-8-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-8-MultiHeadSelfAtte (None, None, 768)    0           Transformer-7-FeedForward-Norm[0]
                                                                 Transformer-8-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-8-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-8-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-8-FeedForward (Feed (None, None, 768)    4722432     Transformer-8-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-8-FeedForward-Dropo (None, None, 768)    0           Transformer-8-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-8-FeedForward-Add ( (None, None, 768)    0           Transformer-8-MultiHeadSelfAttent
                                                                 Transformer-8-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-8-FeedForward-Norm  (None, None, 768)    1536        Transformer-8-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-9-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-8-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-9-MultiHeadSelfAtte (None, None, 768)    0           Transformer-9-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-9-MultiHeadSelfAtte (None, None, 768)    0           Transformer-8-FeedForward-Norm[0]
                                                                 Transformer-9-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-9-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-9-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-9-FeedForward (Feed (None, None, 768)    4722432     Transformer-9-MultiHeadSelfAttent
__________________________________________________________________________________________________
Transformer-9-FeedForward-Dropo (None, None, 768)    0           Transformer-9-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-9-FeedForward-Add ( (None, None, 768)    0           Transformer-9-MultiHeadSelfAttent
                                                                 Transformer-9-FeedForward-Dropout
__________________________________________________________________________________________________
Transformer-9-FeedForward-Norm  (None, None, 768)    1536        Transformer-9-FeedForward-Add[0][
__________________________________________________________________________________________________
Transformer-10-MultiHeadSelfAtt (None, None, 768)    2362368     Transformer-9-FeedForward-Norm[0]
__________________________________________________________________________________________________
Transformer-10-MultiHeadSelfAtt (None, None, 768)    0           Transformer-10-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-10-MultiHeadSelfAtt (None, None, 768)    0           Transformer-9-FeedForward-Norm[0]
                                                                 Transformer-10-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-10-MultiHeadSelfAtt (None, None, 768)    1536        Transformer-10-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-10-FeedForward (Fee (None, None, 768)    4722432     Transformer-10-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-10-FeedForward-Drop (None, None, 768)    0           Transformer-10-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-10-FeedForward-Add  (None, None, 768)    0           Transformer-10-MultiHeadSelfAtten
                                                                 Transformer-10-FeedForward-Dropou
__________________________________________________________________________________________________
Transformer-10-FeedForward-Norm (None, None, 768)    1536        Transformer-10-FeedForward-Add[0]
__________________________________________________________________________________________________
Transformer-11-MultiHeadSelfAtt (None, None, 768)    2362368     Transformer-10-FeedForward-Norm[0
__________________________________________________________________________________________________
Transformer-11-MultiHeadSelfAtt (None, None, 768)    0           Transformer-11-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-11-MultiHeadSelfAtt (None, None, 768)    0           Transformer-10-FeedForward-Norm[0
                                                                 Transformer-11-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-11-MultiHeadSelfAtt (None, None, 768)    1536        Transformer-11-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-11-FeedForward (Fee (None, None, 768)    4722432     Transformer-11-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-11-FeedForward-Drop (None, None, 768)    0           Transformer-11-FeedForward[0][0]
__________________________________________________________________________________________________
Transformer-11-FeedForward-Add  (None, None, 768)    0           Transformer-11-MultiHeadSelfAtten
                                                                 Transformer-11-FeedForward-Dropou
__________________________________________________________________________________________________
Transformer-11-FeedForward-Norm (None, None, 768)    1536        Transformer-11-FeedForward-Add[0]
__________________________________________________________________________________________________
Transformer-12-MultiHeadSelfAtt (None, None, 768)    2362368     Transformer-11-FeedForward-Norm[0
__________________________________________________________________________________________________
Transformer-12-MultiHeadSelfAtt (None, None, 768)    0           Transformer-12-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-12-MultiHeadSelfAtt (None, None, 768)    0           Transformer-11-FeedForward-Norm[0
                                                                 Transformer-12-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-12-MultiHeadSelfAtt (None, None, 768)    1536        Transformer-12-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-12-FeedForward (Fee (None, None, 768)    4722432     Transformer-12-MultiHeadSelfAtten
__________________________________________________________________________________________________
Transformer-12-FeedForward-Drop (None, None, 768)    0           Transformer-12-FeedForward[0][0]
__________________________________________________________________________________________________
gi (InputLayer)                 [(None, 80)]         0
__________________________________________________________________________________________________
Transformer-12-FeedForward-Add  (None, None, 768)    0           Transformer-12-MultiHeadSelfAtten
                                                                 Transformer-12-FeedForward-Dropou
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 80)           6480        gi[0][0]
__________________________________________________________________________________________________
Transformer-12-FeedForward-Norm (None, None, 768)    1536        Transformer-12-FeedForward-Add[0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 80, 1)        0           dense_4[0][0]
__________________________________________________________________________________________________
agn (AGN)                       [(None, 80, 768), (N 0           Transformer-12-FeedForward-Norm[0
                                                                 lambda_1[0][0]
__________________________________________________________________________________________________
lambda (Lambda)                 (None, None, 1)      0           Input-Token[0][0]
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 80, 768)      0           agn[0][0]
                                                                 lambda[0][0]
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, 768)          0           lambda_2[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 768)          0           lambda_3[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 2)            1538        dropout[0][0]
==================================================================================================
Total params: 108,899,666
Trainable params: 108,899,666
Non-trainable params: 0
__________________________________________________________________________________________________
apply fgm
start to fitting...
31/31 [==============================] - 614s 20s/step - loss: 0.9964
Epoch 2/30
30/31 [============================>.] - ETA: 24s - loss: 0.9246- val_acc 0.5520833333333334 - val_f1 0.548951048951049      
new best model, save model to  save2\clf_model.weights...
31/31 [==============================] - 771s 25s/step - loss: 0.9273
Epoch 3/30
30/31 [============================>.] - ETA: 22s - loss: 0.8461- val_acc 0.5416666666666666 - val_f1 0.5355689939527213     
31/31 [==============================] - 718s 23s/step - loss: 0.8453
.
.
.
iteration 1 accuracy: 0.7435416666666666, f1: 0.73400733053290503

Average accuracy: 0.7435416666666666
Average f1: 0.7300733053290503
```
复现图片如下所示
![image](https://github.com/user-attachments/assets/6d64a9f4-fc3e-4abd-8b7a-993d36f6f631)

## 可视化注意力

首先按照上述指令训练一个模型，然后运行 `visualize_attn.py`实现可视化注意力，一个可能的输出如下图所示：
![image](https://github.com/user-attachments/assets/134d5100-c757-4af5-ab92-e89d336c9d19)



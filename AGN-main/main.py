# -*- coding: utf-8 -*-

# 导入必要的库
import os  # 用于文件和目录操作
import sys  # 用于处理命令行参数
import json  # 用于读取和解析 JSON 配置文件
from pprint import pprint  # 用于美观地打印数据结构（如字典）

# 如果环境变量中提供了随机种子值，则设置随机种子以保证实验的可重复性
seed_value = int(os.getenv('RANDOM_SEED', -1))  # 默认种子值为 -1
if seed_value != -1:
    import random  # 用于生成随机数
    random.seed(seed_value)  # 设置 Python 随机数生成器的种子
    import numpy as np  # 用于科学计算
    np.random.seed(seed_value)  # 设置 NumPy 随机数生成器的种子
    import tensorflow as tf  # 用于深度学习模型开发
    tf.set_random_seed(seed_value)  # 设置 TensorFlow 随机数生成器的种子

# 导入自定义模块
from langml.tokenizer import WPTokenizer  # 导入分词器
from dataloader import DataLoader, DataGenerator  # 数据加载和生成器模块
from model import AGNClassifier  # 分类模型
from metrics import Metrics  # 用于评估模型性能的指标模块

# 检查命令行参数是否正确，要求提供配置文件路径
if len(sys.argv) != 2:
    print("usage: python main.py /path/to/config")  # 提示用户使用方法
    exit()  # 退出程序

# 从命令行参数中读取配置文件路径
config_file = str(sys.argv[1])

# 打开并读取配置文件内容
with open(config_file, "r") as reader:
    config = json.load(reader)  # 将配置文件内容加载为字典

# 打印配置内容以供检查
print("config:")
pprint(config)

# 如果保存模型的目录不存在，则创建该目录
if not os.path.exists(config['save_dir']):
    os.makedirs(config['save_dir'])

# 加载分词器
tokenizer = WPTokenizer(os.path.join(config['pretrained_model_dir'], 'vocab.txt'), lowercase=True)
print("successful!")  # 加载成功的提示
tokenizer.enable_truncation(max_length=config['max_len'])  # 设置分词器的最大长度限制

# 加载数据
print("load data...")
dataloader = DataLoader(
    tokenizer,  # 分词器实例
    config['max_len'],  # 数据的最大长度
    use_vae=True,  # 是否使用变分自编码器 (VAE)
    batch_size=config["batch_size"],  # 每批次的数据量
    ae_epochs=config['ae_epochs']  # 变分自编码器的训练轮数
)

# 设置训练集和验证集路径
dataloader.set_train(config['train_path'])  # 设置训练数据路径
dataloader.set_dev(config['dev_path'])  # 设置验证数据路径

# 保存变分自编码器权重和词汇表
dataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder.weights'))
dataloader.save_vocab(os.path.join(config['save_dir'], 'vocab.pickle'))

# 初始化用于记录每次迭代的准确率和 F1 分数的列表
accuracy_list = []
f1_list = []

# 开始多次迭代训练
for idx in range(1, config['iterations'] + 1):  # 迭代次数由配置文件中的 `iterations` 指定
    print("build generator")  # 打印提示信息
    generator = DataGenerator(
        config['batch_size'],  # 批次大小
        config['max_len']  # 最大长度
    )
    generator.set_dataset(dataloader.train_set)  # 设置训练数据集

    # 初始化评估指标回调
    metrics_callback = Metrics(
        config['batch_size'],  # 批次大小
        config['max_len'],  # 最大长度
        dataloader.dev_set,  # 验证数据集
        os.path.join(config['save_dir'], 'clf_model.weights')  # 模型权重保存路径
    )

    # 更新配置以包含生成器的步数和标签大小
    config['steps_per_epoch'] = generator.steps_per_epoch  # 每个 epoch 的步数
    # 保存更新后的 config 到文件
    import json
    with open('config_updated.json', 'w') as f:
        json.dump(config, f, indent=4)
    config['output_size'] = dataloader.label_size  # 输出的标签数量

    # 初始化分类模型
    model = AGNClassifier(config)

    print("start to fitting...")  # 打印提示信息
    # 开始训练模型
    model.model.fit(
        generator.__iter__(),  # 数据生成器
        steps_per_epoch=generator.steps_per_epoch,  # 每个 epoch 的步数
        epochs=config['epochs'],  # 训练的 epoch 数
        callbacks=[metrics_callback],  # 评估回调
        verbose=config['verbose']  # 是否显示详细的训练日志
    )

    # 获取当前迭代的最佳验证集准确率和 F1 分数
    accuracy = max(metrics_callback.history["val_acc"])
    f1 = max(metrics_callback.history["val_f1"])
    accuracy_list.append(accuracy)  # 保存准确率
    f1_list.append(f1)  # 保存 F1 分数

    # 打印当前迭代的结果
    log = f"iteration {idx} accuracy: {accuracy}, f1: {f1}\n"
    print(log)

# 打印所有迭代的平均准确率和 F1 分数
print("Average accuracy:", sum(accuracy_list) / len(accuracy_list))
print("Average f1:", sum(f1_list) / len(f1_list))

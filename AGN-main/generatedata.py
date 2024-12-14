import json
import random

def load_jsonl(file_path):
    """从 JSONL 文件加载数据"""
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data, file_path):
    """保存数据到 JSONL 文件"""
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def create_train_test_data(neg_file, pos_file, train_file, test_file, train_size=10000, test_size=10000):
    # 加载 neg 和 pos 数据
    neg_data = load_jsonl(neg_file)
    pos_data = load_jsonl(pos_file)

    # 打乱数据
    random.shuffle(neg_data)
    random.shuffle(pos_data)

    # 准备训练数据（每一条 neg 和 pos 数据组合为一对）
    train_data = []
    label = 0
    for i in range(train_size // 2):
        neg_sample = neg_data[i]
        pos_sample = pos_data[i]

        # 将 label 按顺序递增
        neg_sample['label'] = label  # neg 的 label
        pos_sample['label'] = label + 1  # pos 的 label

        # 交替添加 neg 和 pos 数据
        train_data.append(neg_sample)
        train_data.append(pos_sample)

        label += 2  # 每次递增 2，确保 neg 和 pos 数据的 label 不重复且递增

    # 准备测试数据（同样交替添加）
    test_data = []
    label = 0
    for i in range(test_size // 2):
        neg_sample = neg_data[i + train_size // 2]  # 使用未用的 neg 数据
        pos_sample = pos_data[i + train_size // 2]  # 使用未用的 pos 数据

        neg_sample['label'] = label  # neg 的 label
        pos_sample['label'] = label + 1  # pos 的 label

        # 交替添加 neg 和 pos 数据
        test_data.append(neg_sample)
        test_data.append(pos_sample)

        label += 2  # 每次递增 2，确保 neg 和 pos 数据的 label 不重复且递增

    # 保存生成的 train 和 test 数据
    save_jsonl(train_data, train_file)
    save_jsonl(test_data, test_file)

# 文件路径
neg_file = 'review_polarity/txt_sentoken/neg.jsonl'  # 替换为实际路径
pos_file = 'review_polarity/txt_sentoken/pos.jsonl'  # 替换为实际路径
train_file = 'SST2/train.jsonl'
test_file = 'SST2/test.jsonl'

# 创建 train 和 test 数据
create_train_test_data(neg_file, pos_file, train_file, test_file)
print("train.jsonl 和 test.jsonl 文件已生成！")

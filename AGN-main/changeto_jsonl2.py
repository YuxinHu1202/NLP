import json

# 读取 sentence_index 和 sentence 对应关系
def load_sentences(sentence_file):
    sentences = {}
    with open(sentence_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过第一行（列名）
        for line in f:
            parts = line.strip().split('\t')
            index = int(parts[0])
            sentence = parts[1]
            sentences[index] = sentence
    return sentences

# 读取 sentence_index 和 label 对应关系
def load_labels(label_file):
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过第一行（列名）
        for line in f:
            parts = line.strip().split(',')
            index = int(parts[0])
            label = parts[1]
            labels[index] = label
    return labels

# 创建 JSONL 数据
def create_jsonl(sentences, labels, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for index in sentences:
            if index in labels:
                data = {
                    "label": labels[index],
                    "text": sentences[index]
                }
                json_line = json.dumps(data, ensure_ascii=False)
                f.write(json_line + '\n')

# 主函数
def main():
    sentence_file = 'stanfordSentimentTreebank\stanfordSentimentTreebank\datasetSentences.txt'  # sentence_index 和 sentence 的文件
    label_file = 'stanfordSentimentTreebank\stanfordSentimentTreebank\datasetSplit.txt'        # sentence_index 和 label 的文件
    output_file = 'stanfordSentimentTreebank\data\sst.jsonl'     # 输出 JSONL 文件的路径

    sentences = load_sentences(sentence_file)
    labels = load_labels(label_file)
    create_jsonl(sentences, labels, output_file)
    print(f"JSONL 文件已生成：{output_file}")

if __name__ == '__main__':
    main()

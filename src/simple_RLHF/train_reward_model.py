#训练回报函数
import sys, os

# 使用注意力来辅助顶层模型分配对各个辅助任务模块的权重
import time

from src.simple_RLHF.PPO import RewardNetwork

path1 = os.getcwd()
path2 = os.path.dirname(path1)
path3 = os.path.dirname(path2)
path4 = os.path.dirname(path3)
path5 = os.path.dirname(path4)

sys.path.append(path2)
sys.path.append(path3)
sys.path.append(path4)
sys.path.append(path5)

import torch

from torch import nn
import json
from src.simple_RLHF.run_time import path_pretrained_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "1, 0, 2, 3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:0"#"cpu"#

#基于人工标注的数据集，以及预训练语言模型，构造用于训练奖励函数的书记。假设：预训练语言模型回答问题的能力比较弱，生成的答案只来行低于人工编写的答案
def build_training_corpus_for_reward_model_training():
    datas = [json.loads(line) for line in open("all20230103_QA_第二期.jsonl", 'r', encoding='utf8').readlines()]
    from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

    tokenizer = BertTokenizer.from_pretrained(path_pretrained_model)
    model = GPT2LMHeadModel.from_pretrained(path_pretrained_model)
    model.to(device)
    text_generator = TextGenerationPipeline(model, tokenizer, device=device)
    # res =  text_generator("这是很久之前的事情了", max_length=100, do_sample=True)

    samples = []
    for j in range(len(datas)):
        print(f"任务进度{j}/{len(datas)}")
        data = datas[j]
        对话 = data["conversations"]
        text = 对话[0]["content"]

        for i in range(2, len(对话), 2):
            question, answer = 对话[i]["content"], 对话[i + 1]["content"]
            prompt = text[:1000] + question + "[分隔符]"
            tokens = tokenizer.tokenize(prompt)
            answer_info_from_model = text_generator(prompt, max_length=len(tokens) + 30, do_sample=True)
            # print("检查数据，提示信息", prompt)
            answer_from_model = answer_info_from_model[0]["generated_text"].split("[分隔符]")[-1]
            # print("检查数据", question, )
            # print("###############")
            samples.append({"prompt": prompt, "positive_answer": answer, "negative_answer": answer_from_model})
    with open("回复正负例数据.jsonl", 'w', encoding='utf8') as f:
        for data in samples:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

from src.simple_RLHF.utils import DataLoader
from torch.optim import AdamW
def train_reward_model():

    def init_model_and_training_plan(model, learning_rate):
        parameters = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters = [
            {"lr": learning_rate, 'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {"lr": learning_rate, 'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(parameters)
        return optimizer


    #加载数据集
    dataloader = DataLoader()
    data_set = dataloader.load_data_set("回复正负例数据.jsonl")[:10]
    print("数据集大小", len(data_set))


    #初始化模型
    lr = 1e-6
    model = RewardNetwork().to(device)
    optimizer = init_model_and_training_plan(model, lr)

    #训练
    for epoch in range(1):
        count = 0
        batch_size=2
        for p_input_ids, p_masks, p_type_ids, n_input_ids, n_masks, n_type_ids in dataloader.iter_data_set(data_set, batch_size, device):
            model.zero_grad()
            reward_p = model.forward(p_input_ids, p_masks)
            reward_n = model.forward(n_input_ids, n_masks)
            loss = -nn.functional.logsigmoid(reward_p - reward_n).mean()
            loss.backward()
            optimizer.step()
            if count%5==0:
                print(f"任务进度epoch:{epoch}, batch序号{count}/{int(len(data_set)/batch_size)}")
                print("损失值", loss)
            count += 1

    #保存模型
    torch.save(model, "reward_model.pth")

    # return F.cross_entropy(pred, labels)

def load_reward_model():
    pass

if __name__ == '__main__':
    # build_training_corpus_for_reward_model_training()
    train_reward_model()


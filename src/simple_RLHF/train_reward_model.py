#训练回报函数
import sys, os

# 使用注意力来辅助顶层模型分配对各个辅助任务模块的权重
import time


path1 = os.getcwd()
path2 = os.path.dirname(path1)
path3 = os.path.dirname(path2)
path4 = os.path.dirname(path3)
path5 = os.path.dirname(path4)

sys.path.append(path2)
sys.path.append(path3)
sys.path.append(path4)
sys.path.append(path5)
from src.simple_RLHF.PPO import RewardNetwork
from src.simple_RLHF.run_time import MAX_LEN_PROMPT, MAX_LEN_RESPONSE
import torch

from torch import nn
import json
from src.simple_RLHF.run_time import path_pretrained_model
from src.simple_RLHF.run_time import PAD_ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "1, 0, 2, 3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:0"#"cpu"#

#基于人工标注的数据集，以及预训练语言模型，构造用于训练奖励函数的书记。假设：预训练语言模型回答问题的能力比较弱，生成的答案只来行低于人工编写的答案
def build_training_corpus_for_reward_model_training():
    datas = [json.loads(line) for line in open("all20230103_QA_第二期.jsonl", 'r', encoding='utf8').readlines()]
    from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

    tokenizer = BertTokenizer.from_pretrained(path_pretrained_model, padding_side='left')
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0
    # tokenizer.eos_token_id = 0

    model = GPT2LMHeadModel.from_pretrained(path_pretrained_model)
    model.to(device)
    text_generator = TextGenerationPipeline(model, tokenizer, device=device, pad_token_id=tokenizer.eos_token_id)
    # res =  text_generator("这是很久之前的事情了", max_length=100, do_sample=True)

    samples = []
    for j in range(len(datas)):
        print(f"任务进度{j}/{len(datas)}")
        data = datas[j]
        对话 = data["conversations"]
        text = 对话[0]["content"]

        for i in range(2, len(对话), 2):
            question, answer = 对话[i]["content"], 对话[i + 1]["content"]
            prompt = (text + question)[:MAX_LEN_PROMPT] + "[分隔符]"
            tokens = tokenizer.tokenize(prompt)
            answer_info_from_model = text_generator(prompt, max_length=len(tokens) + 30, do_sample=True)
            # print("检查数据，提示信息", prompt)
            answer_from_model = answer_info_from_model[0]["generated_text"].split("[分隔符]")[-1].replace(" ", "")
            # print("检查数据", question, )
            # print("###############")
            samples.append({"prompt": prompt, "positive_answer": answer, "negative_answer": answer_from_model})
    with open("回复正负例数据.jsonl", 'w', encoding='utf8') as f:
        for data in samples:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

class PairWiseLoss(nn.Module):
    """Pairwise Loss for Ranking Tasks.

    This PyTorch module computes a pairwise loss for ranking tasks where the
    goal is to compare two inputs and determine which one is "better" than the
    other. Given two input tensors: `chosen_reward` and `reject_reward`, which
    should contain reward values for the "chosen" and "rejected" options,
    respectively, this module computes the probability of the chosen option
    being "better" than the rejected option using a sigmoid function, and then
    takes the negative logarithm of that probability to get the loss. The loss
    is then averaged over the batch dimension and returned as a scalar tensor.
    Note that this module assumes that higher reward values indicate better
    options.
    """

    def __init__(self):
        super(PairWiseLoss, self).__init__()

    def forward(self, chosen_reward: torch.Tensor,
                reject_reward: torch.Tensor) -> torch.Tensor:
        """Compute pairwise loss.

        Args:
        - chosen_reward: A tensor of shape (batch_size,) containing reward values for the chosen option
        - reject_reward: A tensor of shape (batch_size,) containing reward values for the rejected option

        Returns:
        - loss: A scalar tensor containing the computed pairwise loss
        """

        # Compute probability of the chosen option being better than the rejected option
        probs = torch.sigmoid(chosen_reward - reject_reward)

        # Take the negative logarithm of the probability to get the loss
        log_probs = torch.log(probs)
        loss = -log_probs.mean()

        return loss

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
    data_set = dataloader.load_data_set("回复正负例数据.jsonl")
    print("数据集大小", len(data_set))

    #初始化模型
    lr = 1e-6
    model = RewardNetwork().to(device)
    optimizer = init_model_and_training_plan(model, lr)
    model.train()

    #训练
    for epoch in range(5):
        count = 0
        batch_size = 2
        for p_input_ids, p_masks, p_type_ids, n_input_ids, n_masks, n_type_ids in dataloader.iter_data_set(data_set, batch_size, device):
            model.zero_grad()
            reward_p = model.forward(p_input_ids, p_masks)
            reward_n = model.forward(n_input_ids, n_masks)
            # loss = - nn.functional.logsigmoid(reward_p - reward_n).mean()

            chosen_end_scores = []
            rejected_end_scores = []
            c_truncated_reward_list = []
            r_truncated_reward_list = []
            bs = p_input_ids.shape[0]

            prompt_response_masks = p_input_ids > 0.5
            output_end1 = torch.max(prompt_response_masks.sum(1))
            prompt_response_masks = n_input_ids > 0.5
            output_end2 = torch.max(prompt_response_masks.sum(1))
            output_end = max([output_end2, output_end1])
            for i in range(bs):
                # Check if there is any padding otherwise take length of sequence
                c_inds = (p_input_ids[i] == PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else p_input_ids.shape[1]
                r_inds = (n_input_ids[i] == PAD_ID).nonzero()
                r_ind = r_inds[0].item() if len(r_inds) > 0 else n_input_ids.shape[1]
                end_ind = max(c_ind, r_ind)

                # Retrieve first index where trajectories diverge
                divergence_ind = (p_input_ids[i] !=n_input_ids[i]).nonzero()[0]
                assert divergence_ind > 0

                # Index into the correct rewards
                c_truncated_reward = reward_p[i][:output_end]
                r_truncated_reward = reward_n[i][:output_end]
                # print("检查数据", output_end, end_ind)
                # print("c_truncated_reward", c_truncated_reward)
                # print("r_truncated_reward", r_truncated_reward)
                # Append the last rewards to the list of end scores
                chosen_end_scores.append(c_truncated_reward[end_ind-1])
                rejected_end_scores.append(r_truncated_reward[end_ind-1])
                c_truncated_reward_list.append(c_truncated_reward)
                r_truncated_reward_list.append(r_truncated_reward)

            # Stack the end scores and return them
            c_truncated_reward_list = torch.stack(c_truncated_reward_list)
            r_truncated_reward_list = torch.stack(r_truncated_reward_list)
            # Calculate the loss
            loss = PairWiseLoss()(c_truncated_reward_list, r_truncated_reward_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count%5==0:
                print(f"任务进度epoch:{epoch}, batch序号{count}/{int(len(data_set)/batch_size)},损失值{loss}")
            count += 1

    #保存模型
    torch.save(model, "reward_model.pth")

    # return F.cross_entropy(pred, labels)

def load_reward_model():
    pass

if __name__ == '__main__':
    # build_training_corpus_for_reward_model_training()
    train_reward_model()


#使用RLHF策略，训练一个好的信息抽取模型
#参考https://github.com/jianzhnie/open-chatgpt/blob/main/chatgpt/rlhf/ppo_trainer.py
#https://zhuanlan.zhihu.com/p/624914233
#https://zhuanlan.zhihu.com/p/645225982
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
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from torch import nn
from src.simple_RLHF.run_time import path_pretrained_model
import torch
from src.simple_RLHF.utils import DataLoaderPPO
from src.simple_RLHF.PPO import ActorNetwork, RewardNetwork
from torch.optim import AdamW
from collections import OrderedDict
import numpy as np
from src.simple_RLHF.run_time import MAX_LEN_PROMPT, MAX_LEN_RESPONSE



os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "1, 0, 2, 3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:0"#"cpu"#

class ActorCritic(nn.Module):

    def __init__(self, reward_model_path):
        super(ActorCritic, self).__init__()

        # 初始化actor和reference
        self.actor_model = ActorNetwork()
        self.reference_model = ActorNetwork()

        self.actor_optimizer = self.init_training_plan(self.actor_model, 2e-7)

        # 初始化critic和reward,加载微调得到的模型参数
        self.critic_model = RewardNetwork()
        self.reward_model = RewardNetwork()
        if len(reward_model_path)>0:#如果有
            state_dict = torch.load(reward_model_path).state_dict()
            self.critic_model.load_state_dict(state_dict)
            self.reward_model.load_state_dict(state_dict)

        self.critic_optimizer = self.init_training_plan(self.critic_model, 1e-5)

        self.eps = 1e-8
        self.clip_reward_value = 5

    def init_training_plan(self, model, learning_rate):
        parameters = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters = [
            {"lr": learning_rate, 'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {"lr": learning_rate, 'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(parameters)
        return optimizer

    def generate(self, prompt_input_ids, masks, response_starts):
        with torch.no_grad():
            prompt_response_ids = self.actor_model.encoder.generate(prompt_input_ids, max_length=MAX_LEN_PROMPT + MAX_LEN_RESPONSE, return_dict_in_generate=True, pad_token_id=0)[0]
        attention_mask = prompt_response_ids.not_equal(0).long()

        prompt_response_masks = prompt_response_ids > 0.5
        output_ends = prompt_input_ids.shape[1] + prompt_response_masks[:, prompt_input_ids.shape[1]:].sum(1)
        output_ends.unsqueeze(1)

        logits_actor = self.actor_model.forward(prompt_response_ids, attention_mask)
        action_prob = (torch.softmax(logits_actor, dim=-1).max(dim=-1).values)
        log_probs_actor = torch.log(action_prob + self.eps)

        logits_reference = self.reference_model.forward(prompt_response_ids, attention_mask)
        ref_prob = (torch.softmax(logits_reference, dim=-1).max(dim=-1).values)
        log_probs_ref = torch.log(ref_prob + self.eps)
        # 奖励的计算

        kl_ctl = 1e-4
        kl_div = kl_ctl * (log_probs_actor - log_probs_ref)
        rewards = - kl_div
        # print("检查数据")

        rewards_from_env = self.reward_model.forward(prompt_response_ids, prompt_response_masks)
        # print("检查数据rewards_from_env", rewards_from_env)
        reward_clip = torch.clamp(rewards_from_env, -self.clip_reward_value,self.clip_reward_value)

        values = self.critic_model.forward(prompt_response_ids, prompt_response_masks)

        for j in range(response_starts.shape[0]):
            # print("检查数据response_starts[j]:output_ends[j]", response_starts[j], output_end)
            rewards[j, response_starts[j]:output_ends[j]][-1] += reward_clip[j, max(output_ends[j]-1, response_starts[j].item() + 1)]
        # print("检查数据rewards", rewards)
        return prompt_response_ids, prompt_response_masks, logits_actor, log_probs_actor, values, rewards, output_ends

    def forward(self, sequences_actor, sequences_mask_actor, sequences_critic, sequences_mask_critic):
        actor_logits = self.actor_model.forward(sequences_actor, sequences_mask_actor)
        values = self.critic_model.forward(sequences_critic, sequences_mask_critic)
        return actor_logits, values

import torch.nn.functional as F

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def get_advantages_and_returns(values, rewards, starts, ends, gamma=1e-5, lam=9e-1):
    # values（B，L） critic model输出
    # rewards（B，L）reward 包含kl散度
    # start answer开始的位置
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    advantages_reversed = []
    length = rewards.size()[-1]
    # 计算每个时刻（序列位置）的critic model预测误差
    for i in range(starts.shape[0]):
        s, e = starts[i], ends[i]
        lastgaelam = 0
        for t in reversed(range(s, e)):
            nextvalues = values[:, t + 1] if t < e - 1 else 0.0
            # critic model预测的是t到到最后一个时刻的奖励和，所以变化量delta可以用如下公式表示
            delta = (rewards[i, t] + gamma * nextvalues) - values[i, t]
            # self.gamma=1，self.lam=0.95是衰减因子，表示之前计算的delta对现在影响越来越小
            lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, starts:ends]

    # 后续用来更新critic model用
    return advantages.detach(), returns

#策略函数初始状态，直接使用预训练模型
#价值函数初始状态，是使用监督训练得到的
class RLHFTrainer:

    def __init__(self):
        # 初始化模型
        self.actor_critic = ActorCritic("reward_model.pth").to(device)
        self.actor_optimizer, self.critic_optimizer = self.actor_critic.actor_optimizer, self.actor_critic.critic_optimizer
        self.eps = 1e-8
        self.critic_eps_clip = 5
        self.actor_eps_clip = 5
        self.beta_s = 0.01

    def learn(self, memories):
        self.actor_critic.train()
        loss_values = []
        for epoch in range(2):
            for memory in memories:
                values_历史上某个检查点 = memory["values_历史上某个检查点"]
                rewards_历史 = memory["rewards"]
                log_probs_actor_历史 = memory["log_probs_actor_历史"]
                sequences_actor = memory["sequences_actor"]
                sequences_mask_actor = memory["sequences_mask_actor"]
                action_len_actor = memory["action_len_actor"]
                sequences_critic = memory["sequences_critic"]
                sequences_mask_critic = memory["sequences_mask_critic"]
                action_len_critic = memory["action_len_critic"]
                respense_starts = memory["respense_starts"]
                output_ends = memory["output_end"]

                #基于最新的critic计算价值

                logits_actor, values = self.actor_critic.forward(sequences_actor, sequences_mask_actor, sequences_critic, sequences_mask_critic)

                action_prob = (torch.softmax(logits_actor, dim=-1).max(dim=-1).values)
                actions_log_probs = torch.log(action_prob + self.eps)
                kl_div_loss = ((action_prob *(log_probs_actor_历史 - actions_log_probs)).sum( dim=-1).mean())
                entropies = (action_prob * actions_log_probs).sum(dim=-1)

                ratios = (actions_log_probs - log_probs_actor_历史).exp()
                advantages = rewards_历史 - values_历史上某个检查点
                # print("检查数据advantages", advantages)

                # normalize advantages
                advantages = (advantages - advantages.mean(dim=-1).unsqueeze(1)) / ( advantages.std() + self.eps)
                surr1 = advantages * ratios
                surr2 = (torch.clamp(ratios, 1 - self.actor_eps_clip, 1 + self.actor_eps_clip) * advantages)
                policy_loss = -torch.min(surr1, surr2) - self.beta_s * entropies.unsqueeze(1)
                policy_loss = policy_loss.mean()
                loss = policy_loss + kl_div_loss

                value_loss_clipped = values_历史上某个检查点 + (values - values_历史上某个检查点).clamp(-self.critic_eps_clip, self.critic_eps_clip)
                value_loss1 = (value_loss_clipped - rewards_历史)**2
                value_loss2 = (values - rewards_历史)**2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                if torch.isnan(value_loss):
                    raise ValueError('Value loss is nan')
                # print('value_loss', value_loss.item())
                self.actor_critic.zero_grad()
                self.actor_optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.actor_optimizer.step()

                # upate critic with loss
                self.actor_critic.zero_grad()
                self.critic_optimizer.zero_grad()
                value_loss.backward(retain_graph=False)
                self.critic_optimizer.step()
                # print(f"检查数据，actor的loss{loss.item()}, critic的loss{value_loss}")
                loss_values.append(loss.item())
        self.actor_critic.eval()
        print(f"检查数据，actor的loss{np.mean(loss_values)}")

    def train(self):
        # 加载数据集
        dataloader = DataLoaderPPO()
        # data_set = dataloader.load_data_set("回复正负例数据.jsonl")
        # print("数据集大小", len(data_set))
        reply_period = 1

        memories = []
        # 训练
        for epoch in range(1):
            count = 0
            batch_size = 2
            for prompt_ids, masks, type_ids, respense_starts in dataloader.iter_data_set("回复正负例数据.jsonl", batch_size, device):

                #执行一次采样，获取针对当前prompt的输出，并计算相关的价值和奖励
                with torch.no_grad():
                    prompt_response_ids, prompt_response_masks, logits_actor, log_probs_actor, values, rewards, output_ends = self.actor_critic.generate(prompt_ids, masks, respense_starts)

                #收集采样获得的数据
                memories.append({"values_历史上某个检查点": values, "rewards": rewards, "log_probs_actor_历史": log_probs_actor, "respense_starts": respense_starts, "output_end": output_ends,
                                 "sequences_actor": prompt_response_ids, "sequences_mask_actor": prompt_response_masks, "action_len_actor": logits_actor.shape[1],
                                 "sequences_critic": prompt_response_ids, "sequences_mask_critic": prompt_response_masks, "action_len_critic": logits_actor.shape[1]})

                if len(memories)==reply_period:#没个周期学习一次记忆中的最新数据
                    self.learn(memories)
                    memories = []
                count += 1

                if count%20==0:
                    print(f"任务进度{count}/{int(500 / batch_size)}")
                    torch.save(self.actor_critic.actor_model, "actor_model.pth")
        torch.save(self.actor_critic.actor_model, "actor_model.pth")


def test_actor_model():
    tokenizer = BertTokenizer.from_pretrained(path_pretrained_model)
    actor = ActorNetwork()
    state_dict = torch.load("actor_model.pth").state_dict()
    actor.load_state_dict(state_dict)
    actor.eval()
    text_generator = TextGenerationPipeline(actor.encoder, tokenizer)
    prompt = "这是很久之前的事情了"
    prompt = "板说明:\n1.图中未注明板厚为120，板底筋未注明均配双向C10@200,板面筋未注明配贯通钢筋双向C10@200。\n2.板分布筋详见结构设计总说明.\n3.隔墙下结构未布置梁处,板内附加底筋，具体做法详见结构设计总说明.\n4.楼板开洞且洞边砌筑墙,洞口板底筋每边2C14,具体做法详见结构设计总说明.\n5.板端阳角处角部附加筋做法详见结构设计总说明.[SEP]阅读以上建筑设计说明，回答问题。结构类型为板,未注明,面筋[分隔符]"
    res = text_generator(prompt, max_length=MAX_LEN_PROMPT + MAX_LEN_RESPONSE, do_sample=True)
    print(res)
if __name__ == '__main__':

    trainer = RLHFTrainer()
    trainer.train()

    # test_actor_model()

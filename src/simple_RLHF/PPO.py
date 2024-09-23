from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from src.simple_RLHF.run_time import path_pretrained_model
from einops import rearrange
import torch
from collections import OrderedDict

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')
        # mask = mask.repeat(1, 1, 21128)

    masked_seq = seq.masked_fill(mask==0, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

class RewardNetwork(nn.Module):

    def __init__(self, num_binned_output=1):
        super().__init__()
        # 初始化encoder
        self.encoder = GPT2LMHeadModel.from_pretrained(path_pretrained_model)
        self.head = nn.Linear(21128, num_binned_output)

    def forward(self, x, mask):
        embeds = self.encoder(x, attention_mask=mask).logits
        # pooled = masked_mean(embeds, mask, dim=1)
        value = self.head(embeds)
        return value.squeeze(2)

class ActorNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = GPT2LMHeadModel.from_pretrained(path_pretrained_model)
    def forward(self, input_ids, masks):
        # logits = self.encoder.generate(input_ids, attention_mask=masks, max_length=1024)
        logits = self.encoder.forward(input_ids, attention_mask=masks).logits
        return logits

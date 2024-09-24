import json
import random

import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from src.simple_RLHF.run_time import path_pretrained_model
import numpy as np
from src.simple_RLHF.run_time import MAX_LEN_PROMPT, MAX_LEN_RESPONSE
class DataLoader:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(path_pretrained_model)
        self.tokenizer.padding_side = "left"

    def get_features(self, text, max_len=MAX_LEN_PROMPT + MAX_LEN_RESPONSE):
        features = {"text": text}
        tokens = self.tokenizer.tokenize(text)
        features["tokens"] = tokens#["[CLS]"] + tokens
        features["input_ids"] = self.tokenizer.convert_tokens_to_ids(tokens) + [0] * (max_len - len(features["tokens"]))
        features["masks"] = [1] * len(features["tokens"]) + [0] * (max_len - len(features["tokens"]))
        features["type_ids"] = [0] * max_len
        return features

    def load_data_set(self, file_path):
        datas = [json.loads(line) for line in open(file_path, 'r', encoding="utf8").readlines()]

        samples = []
        for data in datas:
            qa_positive = data["prompt"] + data["positive_answer"]
            qa_negative = data["prompt"] + data["negative_answer"]
            samples.append([self.get_features(qa_positive), self.get_features(qa_negative)])

        return samples

    def iter_data_set(self, samples, batch_size, device):
        random.shuffle(samples)
        p_batches, p_batch = [], [[], [], []]
        n_batches, n_batch = [], [[], [], []]
        for sample in samples:
            p, n = sample
            p_batch[0].append(p["input_ids"])
            p_batch[1].append(p["masks"])
            p_batch[2].append(p["type_ids"])

            n_batch[0].append(n["input_ids"])
            n_batch[1].append(n["masks"])
            n_batch[2].append(n["type_ids"])

            if len(p_batch[0])==batch_size:
                torch.tensor(p_batch[0], device=device).long()
                yield torch.tensor(p_batch[0], device=device).long(), torch.tensor(p_batch[1], device=device).long(), torch.tensor(p_batch[2], device=device).long(), \
                      torch.tensor(n_batch[0], device=device).long(), torch.tensor(n_batch[1], device=device).long(), torch.tensor(n_batch[2], device=device).long()
                p_batch, n_batch = [[], [], []], [[], [], []]

        if len(p_batch[0])>0:
            yield torch.tensor(p_batch[0], device=device).long(), torch.tensor(p_batch[1], device=device).long(), torch.tensor(p_batch[2], device=device).long(), \
                  torch.tensor(n_batch[0], device=device).long(), torch.tensor(n_batch[1], device=device).long(), torch.tensor(n_batch[2], device=device).long()
            p_batch, n_batch = [[], [], []], [[], [], []]


class DataLoaderPPO:

    def __init__(self):
        print("检查数据path_pretrained_model", path_pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(path_pretrained_model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def get_features(self, texts, max_len=1024):
        features = {"texts": texts}
        tokens = [self.tokenizer.tokenize(text) for text in texts]
        features["tokens"] = tokens#["[CLS]"] + tokens
        #使用自带工具生成基本数据
        feature = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True)
        features["input_ids"] = feature['input_ids']
        features["masks"] = feature['attention_mask']
        features["type_ids"] = feature['token_type_ids']
        max_seq_len = feature['input_ids'][0].shape[0]
        features["response_starts"] = np.array([[max_seq_len] for _ in range(len(texts))])

        return features

    def load_data_set(self, file_path):
        datas = [json.loads(line) for line in open(file_path, 'r', encoding="utf8").readlines()]
        samples = []
        for data in datas:
            samples.append(self.get_features(data["prompt"]))
        return samples

    def iter_data_set(self, file_path, batch_size, device):
        datas = [json.loads(line) for line in open(file_path, 'r', encoding="utf8").readlines()]

        random.shuffle(datas)
        qa_batch = []
        for data in datas:
            qa_batch.append(data["prompt"])
            if len(qa_batch) == batch_size:
                features = self.get_features(qa_batch)
                yield torch.tensor(features["input_ids"], device=device).long(), torch.tensor(features["masks"],device=device).long(), torch.tensor(features["type_ids"], device=device).long(), torch.IntTensor(features["response_starts"])
                qa_batch = []

        if len(qa_batch) > 0:
            features = self.get_features(qa_batch)
            yield torch.tensor(features["input_ids"], device=device).long(), torch.tensor(features["masks"], device=device).long(), torch.tensor(features["type_ids"], device=device).long(), torch.IntTensor(features["response_starts"])
            qa_batch = []


import json
import random

import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from src.simple_RLHF.run_time import path_pretrained_model

class DataLoader:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(path_pretrained_model)

    def get_features(self, text, max_len=1024):
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

    def get_features(self, text, max_len=1000):
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
            samples.append(self.get_features(data["prompt"]))
        return samples

    def iter_data_set(self, samples, batch_size, device):
        random.shuffle(samples)
        batches, batch = [], [[], [], [], [], []]
        for sample in samples:
            batch[0].append(sample["input_ids"])
            batch[1].append(sample["masks"])
            batch[2].append(sample["type_ids"])
            batch[3].append(len(sample["tokens"]))

            if len(batch[0])==batch_size:
                torch.tensor(batch[0], device=device).long()
                yield torch.tensor(batch[0], device=device).long(), torch.tensor(batch[1], device=device).long(), torch.tensor(batch[2], device=device).long(), torch.IntTensor(batch[3])
                batch = [[], [], [], [], []]

        if len(batch[0])>0:
            yield torch.tensor(batch[0], device=device).long(), torch.tensor(batch[1], device=device).long(), torch.tensor(batch[2], device=device).long(), torch.IntTensor(batch[3])
            batch = [[], [], [], [], []]

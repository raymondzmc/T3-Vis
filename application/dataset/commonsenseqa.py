import os
import torch
import copy

from datasets import load_dataset
from transformers import AutoTokenizer

import pdb



class Dataset(object):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return self.data

    def __getitem__(self, index):
        return self.data[index]


def preprocess_function_abs(tokenizer, doc):
    return tokenizer(doc, truncation=True, padding="longest", return_tensors="pt")

def commonsenseqa_test_set():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gold_dir = 'resources/gpt2-metaicl-commonsenseqa/out/channel-metaicl-gold'
    random_dir = 'resources/gpt2-metaicl-commonsenseqa/out/channel-metaicl-random'
    gold_data = torch.load(os.path.join(gold_dir, 'tensorized_inputs.pt'))
    random_data = torch.load(os.path.join(random_dir, 'tensorized_inputs.pt'))

    gold_data = [{'input_ids': gold_data['input_ids'], 'attention_mask': gold_data['attention_mask'], 'token_type_ids': gold_data['token_type_ids']} for i in range(500)]
    random_data = [{'input_ids': random_data['input_ids'], 'attention_mask': random_data['attention_mask'], 'token_type_ids': random_data['token_type_ids']} for i in range(500)]
    data = gold_data + random_data
    # # gold_data = [{''}]
    # pdb.set_trace()
    # data = Dataset()


    # with open('../resources/opus-2020-07-17.test.txt', 'r') as f:
    #     lines = f.readlines()

    # dataset = []
    # for line in lines:
    #     if line.startswith('>>yue_Hant<<'):
    #         dataset.append(line)

    # # visualize_columns = dataset.column_names
    # tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    # data = [preprocess_function_abs(tokenizer, x) for x in dataset]

    dataset = Dataset(data)
    # setattr(dataset, 'visualize_columns', visualize_columns)
    setattr(dataset, 'input_columns', ['input_ids', 'attention_mask', 'token_type_ids'])
    # setattr(dataset, 'target_columns', ['highlights'])
    setattr(dataset, 'max_length', 1024)
    setattr(dataset, 'tokenizer', tokenizer)
    return dataset
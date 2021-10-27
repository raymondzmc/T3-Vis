import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pdb

# class BertClassification(nn.Module):
#     """
#     Wrapper class for BertForSequenceClassification
#     """
#     def __init__(self):
#         super(BertClassification, self).__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#         self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
#         self.hidden_size = self.model.config.hidden_size
#         self.num_hidden_layers = self.model.config.num_hidden_layers
#         self.num_attention_heads = self.model.config.num_hidden_layers

#     def prune_heads(heads_to_prune):
#         self.model.prune_heads(heads_to_prune)

#     def forward(self, example, **kwargs):
#         model_input = self.format_for_input(example)
#         out = self.model(**model_input, **kwargs)
#         return out


def bert_classifier():
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    setattr(model, 'num_hidden_layers', model.config.num_hidden_layers)
    setattr(model, 'num_attention_heads', model.config.num_attention_heads)
    setattr(model, 'hidden_size', model.config.hidden_size)
    return model
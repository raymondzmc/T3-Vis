import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel, AutoTokenizer

from collections import defaultdict 
import pdb


class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x).squeeze(-1)
        out = self.sigmoid(x)
        return out

class BERT_SUM(nn.Module):
    def __init__(self):
        super(BERT_SUM, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Transformer document encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = Classifier(self.bert.config.hidden_size)


        # Initialize the parameters of summarization specific layers if a checkpoint is not loaded 
        for p in self.ext_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, seg_ids, input_mask, cls_idx, cls_mask, sent_labels=None, head_mask=None, output_attentions=False):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        encoder_output = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=seg_ids, head_mask=head_mask, output_attentions=output_attentions)
        sent_vectors = hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), cls_idx]
        sent_vectors = sent_vectors * cls_mask[:, :, None].float()

        sent_scores = (self.ext_layer(span_vectors) * cls_mask.float()).squeeze(-1)

        if output_attentions:
            output = (sent_scores, encoder_output[-1])
        else:
            output = span_scores

        return output


def bert_summarizer():
    model = BERT_SUM()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer
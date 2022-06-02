import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import pdb

def longformer_finetuned_papers():
    model = AutoModelForSequenceClassification.from_pretrained('danielhou13/longformer-finetuned_papers', num_labels = 2)
    setattr(model, 'num_hidden_layers', model.config.num_hidden_layers)
    setattr(model, 'num_attention_heads', model.config.num_attention_heads)
    setattr(model, 'hidden_size', model.config.hidden_size)
    return model
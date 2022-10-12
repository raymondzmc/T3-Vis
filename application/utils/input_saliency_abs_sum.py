# https://github.com/pytorch/captum/blob/4378c1c7ae733a56fa781881faa3cf59b8829b6b/tutorials/seq2seq_interpret.ipynb

import torch
import torch.nn as nn
import numpy as np

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers.models.pegasus.modeling_pegasus import PegasusSinusoidalPositionalEmbedding, \
    PegasusEncoderLayer, PegasusDecoderLayer, PegasusAttention

from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail').model
interpretable_embedding = configure_interpretable_embedding_layer(model, 'encoder.embed_tokens')


def ig_forward(src, trg, idx):
    return model.forward(src, trg)[idx][0].unsqueeze(0)


def compute_input_saliency(model, input_ids, output_ids):
    model.eval()
    model.zero_grad()

    # pre-computing word embeddings
    # src_indices = torch.LongTensor(input_ids)
    # src_embedding = interpretable_embedding.indices_to_embeddings(input_ids)

    embeddings = model.model.decoder.embed_tokens.weight.data  # vocab_size * emb_dim
    src_embedding = torch.index_select(embeddings, 0, input_ids)  # X = input_len * emb
    # print(src_embedding)
    # print(src_embedding.size())
    # print(output_ids)
    # outputs = model(src_embedding, output_ids, 0).squeeze(0)
    # print("output ids", type(output_ids))
    # print(outputs)
    saliency = {
        # 'lrp': [],
        # 'inputGrad': [],
        'integratedGrad': [],
    }
    attribution_igs = []  # size: [len(output_tokens), len(src_tokens)]
    ig = IntegratedGradients(ig_forward)
    for idx, output_id in enumerate(output_ids):
        max_idx = max(output_id)
        # print(max_idx)
        attribution_ig = ig.attribute(inputs=src_embedding,
                                      additional_forward_args=(torch.LongTensor(output_ids), idx),
                                      target=max_idx)
        # print("ig_attributions", attribution_ig)
        attribution_igs.append(attribution_ig)

    model.zero_grad()
    return saliency


if __name__ == "__main__":
    # (not used)
    # Example code
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
    model.train()

    # Input x Gradients

    # register_hooks(model)

    batch_size = 0
    inputs = tokenizer("There is an cat", return_tensors="pt")
    outputs = model(**inputs)

    pdb.set_trace()

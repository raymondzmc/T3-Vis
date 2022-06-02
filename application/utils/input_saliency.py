import math
import torch
import torch.nn as nn
import numpy as np
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers.models.longformer.modeling_longformer import LongformerEmbeddings, LongformerLayer, LongformerSelfAttention

import pdb

def normalize_tensor(tensor):
    normalized = (tensor - tensor.min()) /\
                 (tensor.max() - tensor.min())
    return normalized

def rescale(out_relevance, inp_relevances, epsilon=1e-7):
    inp_relevances = torch.abs(inp_relevances)
    if len(out_relevance.shape) == 2:
        ref_scale = torch.sum(out_relevance, dim=-1, keepdim=True) + epsilon
        inp_scale = torch.sum(inp_relevances, dim=-1, keepdim=True) + epsilon
    elif len(out_relevance.shape) == 3:
        ref_scale = out_relevance.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + epsilon
        inp_scale = inp_relevances.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + epsilon
    scaler = ref_scale / inp_scale
    inp_relevances = inp_relevances * scaler
    return inp_relevances

def lrp_linear(w, b, output, out_relevance, alpha=0.5, beta=0.5, epsilon=1e-7):

    # Positive contributions
    w_p = torch.clamp(w, min=0.0)
    b_p = torch.clamp(b, min=0.0)
    z_p = torch.matmul(output, w_p.T) + b_p + epsilon
    s_p = out_relevance / z_p
    c_p = torch.matmul(s_p, w_p)
    
    # Negative contributions
    w_n = torch.clamp(w, max=0.0)
    b_n = torch.clamp(b, max=0.0)
    z_n = torch.matmul(output, w_n.T) + b_n - epsilon
    s_n = out_relevance / z_n
    c_n = torch.matmul(s_n, w_n)

    inp_relevance = output * (alpha * c_p + beta * c_n)
    
    out_relevance = rescale(out_relevance, inp_relevance)

    return out_relevance



def register_hooks(model):
    allowed_pass_layers = (
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.Tanh,
        LongformerEmbeddings,
    )

    def forward_hook(module, input, output):
        setattr(module, 'input', [x for x in input])
        try:
            setattr(module, 'activation', output)
        except:
            pass

    def forward_hook_attn(module, input, output):
        setattr(module, 'input', [x for x in input])
        setattr(module, 'activation', [x for x in output])


    for name, module in model.named_children():
        if isinstance(module, LongformerSelfAttention):
            module.register_forward_hook(forward_hook_attn)
        elif not isinstance(module, allowed_pass_layers):
            module.register_forward_hook(forward_hook)

        register_hooks(module)


def bert_layer_lrp(layer, relevance):


    if relevance.shape != layer.output.activation.shape:
        relevance_holder = torch.zeros(layer.output.activation.shape)
        relevance_holder[:, 0] = relevance # Since only the CLS token is used for classfication (Could also be repeated)
        relevance = relevance_holder
        del relevance_holder


    relevance = relevance.to(next(layer.parameters()).device)


    # Relevance for the FF sub-layer
    relevance_residual_ff = torch.autograd.grad(layer.output.activation, layer.output.input[1], grad_outputs=relevance, retain_graph=True, allow_unused=True)[0]

    relevance = torch.autograd.grad(layer.output.activation, layer.output.dense.activation, grad_outputs=relevance, retain_graph=True, allow_unused=True)[0]
    relevance = lrp_linear(layer.output.dense.weight, layer.output.dense.bias, layer.output.dense.input[0], relevance)
    relevance = lrp_linear(layer.intermediate.dense.weight, layer.intermediate.dense.bias, layer.intermediate.dense.input[0], relevance)
    relevance += relevance_residual_ff

    # Relevance for the self-attention output transformation
    relevance_residual_attn = torch.autograd.grad(layer.attention.output.activation, layer.attention.output.input[1], grad_outputs=relevance, retain_graph=True, allow_unused=True)[0]

    relevance = torch.autograd.grad(layer.attention.output.activation, layer.attention.output.dense.activation, grad_outputs=relevance, retain_graph=True, allow_unused=True)[0]
    relevance = lrp_linear(layer.attention.output.dense.weight, layer.attention.output.dense.bias, layer.attention.output.dense.input[0], relevance)

    # Relevance for the self-attention mechanism
    self_attention = layer.attention.self
    key, query, value = self_attention.key, self_attention.query, self_attention.value
    relevance_key = torch.autograd.grad(self_attention.activation[0], key.activation,  grad_outputs=relevance,  retain_graph=True, allow_unused=True)[0]
    relevance_query = torch.autograd.grad(self_attention.activation[0], query.activation,  grad_outputs=relevance,  retain_graph=True, allow_unused=True)[0]
    relevance_value = torch.autograd.grad(self_attention.activation[0], value.activation,  grad_outputs=relevance,  retain_graph=True, allow_unused=True)[0]

    relevance_key = lrp_linear(key.weight, key.bias, key.input[0], relevance)
    relevance_query = lrp_linear(query.weight, query.bias, query.input[0], relevance)
    relevance_value = lrp_linear(value.weight, value.bias, value.input[0], relevance)

    relevance = relevance_query + relevance_key + relevance_value

    relevance += relevance_residual_attn
    return relevance


def bert_lrp(model, out_relevance, grad=None):
    """
    Recursively computes the LRP score given the model and prediction

    TO DO: Make this function into a class similar to the implemenation from
    https://github.com/moboehle/Pytorch-LRP/blob/master/inverter_util.py
    """
    allowed_pass_layers = (
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.Tanh,
        BertEmbeddings,
    ) 

    module_list = []

    # Invert the module list
    module_list = [_ for _ in model.named_children()][::-1]
    relevance = out_relevance


    for name, module in module_list:
        if name == 'classifier':
            relevance = lrp_linear(module.weight, module.bias, module.input[0], relevance)
        elif name == 'pooler':
            relevance = lrp_linear(module.dense.weight, module.dense.bias, module.dense.input[0], relevance)
            # relevance_all = torch.zeros_like(module.input[0])
            # relevance_all[:, 0] = relevance
        elif isinstance(module, LongformerLayer):
            relevance = bert_layer_lrp(module, relevance)
        elif isinstance(module, allowed_pass_layers):
            continue
        else:
            relevance = bert_lrp(module, relevance, grad)

    return relevance


def compute_input_saliency(model, input_len, logits):
    model.zero_grad()

    dim = 1 # Dimension for data

    output_len = logits.size(dim)

    saliency = {
        'lrp': [],
        'inputGrad': [],
    }

    for i in range(output_len):
        model.zero_grad()
        prediction_mask = torch.zeros_like(logits)
        prediction_mask[:, i] = 1
        out_relevance = logits * prediction_mask

        relevance = bert_lrp(model, out_relevance).sum(-1).squeeze(0).detach().abs()
        saliency['lrp'].append(normalize_tensor(relevance).tolist())

        embedding_output = model.bert.embeddings.word_embeddings.activation

        # Replace this hard-code line later
        grad = torch.autograd.grad(out_relevance.sum(), embedding_output, retain_graph=True)[0]
        embeddings = embedding_output
        grad_input = (grad * embeddings).sum(-1).squeeze(0).detach().abs()
        saliency['inputGrad'].append(normalize_tensor(grad_input).tolist())

    model.zero_grad()
    return saliency






if __name__ == "__main__":

    # Example code
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
    model.train()


    # Input x Gradients

    register_hooks(model)
 

    batch_size = 0
    inputs = tokenizer("There is an cat", return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

    # LRP with respection the prediction
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits
    prediction_mask = torch.zeros(logits.size())
    prediction_mask[0, 0] = 1  # Pretend this is the prediction
    out_relevance = logits * prediction_mask
    relevance = bert_lrp(model, out_relevance) # Relevance across all dimensions of the embeddings
    pdb.set_trace()

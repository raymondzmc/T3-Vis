import torch
import numpy as np
from tqdm import tqdm
import pdb
from captum.attr import IntegratedGradients, LayerIntegratedGradients


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    List, int, int, set -> Tuple[set, "torch.LongTensor"]
    """

    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0

    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[~mask].long()
    return heads, index


def compute_taylor_importance_score(model, attn, layer_index, n_heads, head_size, importance_scores,
                                    input_embeddings=None):
    # pruned_heads = attn.pruned_heads
    pruned_heads = set()
    leftover_heads = set(list(range(n_heads))) - pruned_heads

    for head_idx in leftover_heads:
        heads, index = find_pruneable_heads_and_indices([head_idx], n_heads, head_size, pruned_heads)
        # print("heads, index ", heads, index)
        index = index.to(model.device)

        attn.q_proj.bias.sum().backward()
        attn.k_proj.bias.sum().backward()
        attn.v_proj.bias.sum().backward()
        attn.q_proj.weight.sum().backward()
        attn.k_proj.weight.sum().backward()
        attn.v_proj.weight.sum().backward()
        attn.out_proj.weight.sum().backward()

        query_b_grad = (attn.q_proj.bias.grad[index] * attn.q_proj.bias[index]) ** 2
        query_W_grad = (attn.q_proj.weight.grad.index_select(0, index) *
                        attn.q_proj.weight.index_select(0, index)) ** 2

        key_b_grad = (attn.k_proj.bias.grad[index] *
                      attn.k_proj.bias[index]) ** 2
        key_W_grad = (attn.k_proj.weight.grad.index_select(0, index) *
                      attn.k_proj.weight.index_select(0, index)) ** 2

        value_b_grad = (attn.v_proj.bias.grad[index] *
                        attn.v_proj.bias[index]) ** 2
        value_W_grad = (attn.v_proj.weight.grad.index_select(0, index) *
                        attn.v_proj.weight.index_select(0, index)) ** 2

        output_W_grad = (attn.out_proj.weight.grad.index_select(1, index) *
                         attn.out_proj.weight.index_select(1, index)) ** 2
        abs_grad_magnitude = query_b_grad.sum() + query_W_grad.sum() + key_b_grad.sum() + key_W_grad.sum() + \
                             value_b_grad.sum() + value_W_grad.sum() + output_W_grad.sum()

        importance_scores[layer_index, head_idx] += abs_grad_magnitude

    return importance_scores


def compute_ig_importance_score(model, attn, layer_index, n_heads, head_size, importance_scores, input_embeddings):
    ig = IntegratedGradients(model)
    # print("input_embeddings shape", input_embeddings.shape)
    for head_idx in set(list(range(n_heads))):
        heads, index = find_pruneable_heads_and_indices([head_idx], n_heads, head_size, set())
        # print("heads, index", heads, index)
        Wh_Q = attn.q_proj.weight.index_select(0, index)  # W_h^Q  d * d_q
        Wh_K = attn.k_proj.weight.index_select(0, index)  # W_h^K  d * d_k
        # print(Wh_Q.shape, Wh_K.shape)
        Q = torch.matmul(input_embeddings, Wh_Q.T)  # N * d_q
        K = torch.matmul(input_embeddings, Wh_K.T)  # N * d_k
        # print(Q.shape, K.shape)
        A_h = torch.nn.functional.softmax(torch.matmul(Q, K.T) / K.shape[1], dim=0)
        A_baseline = torch.zeros_like(A_h)
        # print(A_h)
        importance_scores[layer_index, head_idx] += ig.attribute(inputs=A_h, baselines=A_baseline,
                                                                 # target=
                                                                 )

    return importance_scores


def get_head_importance_pegasus(model, method='taylor', input_ids=None):
    methods = ['taylor', 'ig']
    if method not in methods:
        raise NotImplementedError(f"Attribution method '{method}' for attention is not yet supported")

    configuration = model.config
    n_layers_encoder = configuration.encoder_layers
    n_heads_encoder = configuration.encoder_attention_heads
    n_layers_decoder = configuration.decoder_layers
    n_heads_decoder = configuration.decoder_attention_heads
    # print("vocab", configuration.vocab_size)
    head_size_encoder = int(configuration.d_model / n_heads_encoder)
    head_size_decoder = int(configuration.d_model / n_heads_decoder)

    importance_scores_encoder = np.zeros((n_layers_encoder, n_heads_encoder))  # N * N
    importance_scores_decoder = np.zeros((n_layers_decoder, n_heads_decoder))  # M * M
    importance_scores_cross = np.zeros((n_layers_decoder, n_heads_encoder))  # M * N

    attribution_functions = [compute_taylor_importance_score, compute_ig_importance_score]
    attribution_fn = attribution_functions[0] if method == 'taylor' else attribution_functions[1]

    input_embeddings = []
    if method == 'ig':
        embeddings = model.model.decoder.embed_tokens.weight.data  # vocab_size * emb_dim
        input_embeddings = torch.index_select(embeddings, 0, input_ids)  # X = input_len * emb

    if method == 'taylor':
        for i in range(n_layers_encoder):
            self_attention_encoder = model.model.encoder.layers[i].self_attn
            # num_attention_heads_encoder = self_attention_encoder.num_heads
            importance_scores_encoder = attribution_fn(model, self_attention_encoder, i,
                                                       n_heads_encoder, head_size_encoder,
                                                       importance_scores_encoder)
    for i in range(n_layers_decoder):
        self_attention_decoder = model.model.decoder.layers[i].self_attn
        cross_attention = model.model.decoder.layers[i].encoder_attn
        importance_scores_decoder = attribution_fn(model, self_attention_decoder, i,
                                                   n_heads_decoder, head_size_decoder,
                                                   importance_scores_decoder, input_embeddings)
        importance_scores_cross = attribution_fn(model, cross_attention, i, n_heads_encoder,
                                                 head_size_encoder, importance_scores_cross, input_embeddings)

    return {"encoder": importance_scores_encoder,
            "decoder": importance_scores_decoder,
            "cross": importance_scores_cross}


# def get_taylor_importance(model):
#     n_layers = model.num_hidden_layers
#     n_heads = model.num_attention_heads
#     head_size = int(model.hidden_size / n_heads)
#     importance_scores = np.zeros((n_layers, n_heads))
#
#     for i in range(n_layers):
#         attention = model.bert.encoder.layer[i].attention
#         num_attention_heads = attention.self.num_attention_heads
#
#         pruned_heads = attention.pruned_heads
#         leftover_heads = set(list(range(n_heads))) - pruned_heads
#
#         for head_idx in leftover_heads:
#             heads, index = find_pruneable_heads_and_indices([head_idx], num_attention_heads, head_size, pruned_heads)
#             index = index.to(model.device)
#
#             query_b_grad = (attention.self.query.bias.grad[index] *\
#                             attention.self.query.bias[index]) ** 2
#             query_W_grad = (attention.self.query.weight.grad.index_select(0, index) *\
#                             attention.self.query.weight.index_select(0, index)) ** 2
#
#             key_b_grad = (attention.self.key.bias.grad[index] *\
#                           attention.self.key.bias[index]) ** 2
#             key_W_grad = (attention.self.key.weight.grad.index_select(0, index) *\
#                           attention.self.key.weight.index_select(0, index)) ** 2
#
#             value_b_grad = (attention.self.value.bias.grad[index] *\
#                             attention.self.value.bias[index]) ** 2
#             value_W_grad = (attention.self.value.weight.grad.index_select(0, index) *\
#                             attention.self.value.weight.index_select(0, index)) ** 2
#
#             output_W_grad = (attention.output.dense.weight.grad.index_select(1, index) *
#                              attention.output.dense.weight.index_select(1, index)) ** 2
#             abs_grad_magnitude = query_b_grad.sum() + query_W_grad.sum() + key_b_grad.sum() + \
#                 key_W_grad.sum() + value_b_grad.sum() + value_W_grad.sum() + output_W_grad.sum()
#
#
#             score = abs_grad_magnitude.item()
#             importance_scores[i, head_idx] += score
#     return importance_scores


def compute_importance(model, dataloader, measure='taylor'):
    assert measure in ['taylor', 'oracle', 'sensitivity']

    max_input_len = model.bert.config.max_position_embeddings
    n_layers = model.bert.config.num_hidden_layers
    n_heads = model.bert.config.num_attention_heads
    head_size = int(model.bert.config.hidden_size / n_heads)

    importance_scores = np.zeros((n_layers, n_heads))

    device = model.device
    total_loss = 0.

    if measure == 'sensitivity':
        head_mask = torch.ones(n_layers, n_heads).to(device)
        head_mask.requires_grad_(requires_grad=True)
    else:
        head_mask = None

    for step, inputs in enumerate(tqdm(dataloader)):
        batch_size_ = inputs['input_ids'].__len__()

        if torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()

        output = model(**inputs)
        loss = output['loss']
        loss.backward()

        if measure == 'sensitivity':
            importance_scores += head_mask.grad.abs().detach().cpu().numpy()
        elif measure == 'taylor':
            importance_scores = get_taylor_importance(model)

    return importance_scores

import torch
import numpy as np

def normalize(matrix, axis=None):
    normalized = (matrix - matrix.min(axis=axis)) /\
                 (matrix.max(axis=axis) - matrix.min(axis=axis))
    return normalized

def format_attention_image(attention):
    formatted_attn = []
    for layer_idx in range(attention.shape[0]):
        for head_idx in range(attention.shape[1]):
            formatted_entry = {
                'layer': layer_idx,
                'head': head_idx
            }


            # Flatten value of log attention normalize between 255 and 0
            if len(attention[layer_idx, head_idx]) == 0:
                continue
            attn = np.array(attention[layer_idx, head_idx]).flatten()
            attn = (attn - attn.min()) / (attn.max() - attn.min())
            alpha = np.round(attn * 255)
            red = np.ones_like(alpha) * 255
            green = np.zeros_like(alpha) * 255
            blue = np.zeros_like(alpha) * 255

            attn_data = np.dstack([red,green,blue,alpha]).reshape(alpha.shape[0] * 4).astype('uint8')
            formatted_entry['attn'] = attn_data.tolist()
            formatted_attn.append(formatted_entry)
    return formatted_attn


def format_attention(output_attention, n_heads, pruned_heads):
    attentions = [(l.squeeze(0) * 100).round().byte().cpu() for l in output_attention]
    attn_vectors = []
    for layer in range(len(attentions)):
        attn_vectors.append([])
        next_head_idx = 0
        for head in range(n_heads):
            if (layer in pruned_heads.keys()) and (head in pruned_heads[layer]):
                attn_vectors[layer].append([])
            else:
                attn_vectors[layer].append(attentions[layer][next_head_idx].tolist())
                next_head_idx += 1
    return attn_vectors
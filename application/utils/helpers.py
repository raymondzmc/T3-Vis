import pdb
import torch
import numpy as np
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

def normalize(matrix, axis=None):
    normalized = (matrix - matrix.min(axis=axis)) /\
                 (matrix.max(axis=axis) - matrix.min(axis=axis))
    return normalized

def format_attention_image(attention, color='red'):
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

            if color == 'red':
                red = np.ones_like(alpha) * 255
                green = np.zeros_like(alpha) * 255
                blue = np.zeros_like(alpha) * 255
            elif color == 'blue':
                blue = np.ones_like(alpha) * 255
                green = np.zeros_like(alpha) * 255
                red = np.zeros_like(alpha) * 255

            attn_data = np.dstack([red,green,blue,alpha]).reshape(alpha.shape[0] * 4).astype('uint8')
            formatted_entry['attn'] = attn_data.tolist()
            formatted_attn.append(formatted_entry)
    return formatted_attn

def compute_aggregated_attn(model, dataloader, max_input_len, measure='taylor'):

    n_layers = model.bert.config.num_hidden_layers
    n_heads = model.bert.config.num_attention_heads
    head_size = int(model.bert.config.hidden_size / n_heads)
    n_examples = len(dataloader.dataset)

    # importance_scores = np.zeros((n_layers, n_heads))

    device = model.device
    total_loss = 0.
    attn = np.zeros((n_layers, n_heads, max_input_len, max_input_len))
    model.eval()

    attn_normalize_count = torch.zeros(max_input_len, device=device)

    for step, inputs in enumerate(tqdm(dataloader)):

        batch_size_ = inputs['input_ids'].__len__()

        if torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()
                    
        inputs['output_attentions'] = True

        with torch.no_grad():
            output = model(**inputs)
        attn_normalize_count += inputs['attention_mask'].sum(dim=0)

        batch_attn = output[-1]
        batch_attn = torch.cat([l.sum(dim=0).unsqueeze(0) for l in batch_attn], dim=0).cpu().numpy()

        attn += batch_attn

    max_input_len = len(attn_normalize_count.nonzero(as_tuple=False))
    attn = attn[:, :, :max_input_len, :max_input_len]
    attn /= attn_normalize_count.cpu().numpy()[:max_input_len]
    formatted_attn = format_attention_image(attn)
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

# def format_attention(attentions):
#     attentions = (torch.stack(attentions).squeeze(1) * 100).byte().tolist()
#     pdb.set_trace()




def output_hidden(model, dataloader, layers=None, max_entries=5000):
    tsne_model = TSNE(n_components=2,
                      verbose=0,
                      perplexity=30,
                      learning_rate='auto',
                      n_iter=2000,
                      init='random',
                      metric='precomputed',
                      random_state=0,
                      square_distances=True)

    model.eval()

    num_layers = model.num_hidden_layers + 1

    hidden_states = torch.zeros(len(dataloader.dataset), num_layers, 768)
    tsne_vectors = np.zeros((max_entries, num_layers, 2))
    labels = np.zeros(len(dataloader.dataset))

    for step, inputs in enumerate(tqdm(dataloader)):
        batch_size_ = inputs['input_ids'].__len__()

        if torch.cuda.is_available():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()

        inputs['output_hidden_states'] = True

        idx = inputs['idx'].cpu().tolist()
        del inputs['idx']
        labels[idx] = inputs['labels'].cpu()

        # The first token is used for classification
        output = model(**inputs)
        for i in range(len(output[2])):
            hidden_states[idx, i] = output[2][i][:, 0, :].detach().cpu()

    for i in range(num_layers):
        hidden_states_ = hidden_states[:max_entries, i].numpy()
        distance = pairwise_distances(hidden_states_, hidden_states_, metric='cosine', n_jobs=4)
        tsne_vectors_ = tsne_model.fit_transform(distance).round(decimals=5)
        tsne_vectors[:, i] = tsne_vectors_

    return tsne_vectors, labels[:max_entries]
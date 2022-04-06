import os
import json
import torch
import flask
import pickle
import argparse
import collections

import numpy as np
from os.path import join as pjoin

from utils.input_saliency import register_hooks, compute_input_saliency
from utils.head_importance import get_taylor_importance
from utils.helpers import normalize, format_attention, format_attention_image

import pdb




################### Global Objects ###################
app = flask.Flask(__name__, template_folder='templates')


class T3_Visualization(object):
    """
    Visualization class for tracking the backend state as well as methods for processing data
    """

    def __init__(self, args):

        self.args = args

        try:
            exec(f"from dataset import {args.dataset}")
        except ImportError:
            print(f"\nWarning: Cannot import function \"{args.dataset}\" from directory \"dataset\", please ensure the function is defined in this file!")

        self.resource_dir = args.resource_dir
        self.checkpoint_dirs = [subdir for subdir in os.listdir(self.resource_dir) if os.path.isdir(pjoin(self.resource_dir, subdir))]
        self.curr_checkpoint_dir = None

        self.filter_paddings = args.filter_paddings

        # TO DO: Set the pretrained model as an attribute of the model to get called

        self.init_model(args.model)
        self.dataset = eval(f"{args.dataset}()")
        self.num_hidden_layers = self.model.num_hidden_layers
        self.num_attention_heads = self.model.num_attention_heads
        self.pruned_heads = collections.defaultdict(list)


        self.table_headings = tuple(self.dataset.visualize_columns)
        self.table_content = [{col_name:row[col_name] for col_name in self.table_headings} for i, row in enumerate(self.dataset) if (args.n_examples and i < args.n_examples)]

    def init_model(self, model_name):
        try:
            exec(f"from models import {args.model}")
        except ImportError:
            print(f"\nWarning: Cannot import function \"{args.model}\" from directory \"models\", please ensure the function is defined in this file!")

        self.model = eval(f"{model_name}()")

        if self.curr_checkpoint_dir != None:
            self.model.load_state_dict(torch.load(pjoin(curr_checkpoint_dir, 'model.pt')))

    def prune_heads(self, heads_to_prune):
        if heads_to_prune == {} and self.pruned_heads != {}:
            self.init_model(self.args.model)
        else:
            for layer, heads in heads_to_prune.items():
                pruned_heads = self.pruned_heads[layer]
                heads_to_prune[layer] = list(set(heads) - set(pruned_heads))
                self.pruned_heads[layer] = set(heads + pruned_heads)
            self.model.prune_heads(heads_to_prune)

    def evaluate_example(self, idx):
        """
        Perform inference on a single data example,
        return output logits, attention scores, saliency maps along with other attributes 
        """
        results = {}

        self.model.train()
        self.model.zero_grad()
        register_hooks(self.model)

        example = self.dataset[idx]

        for col in self.dataset.input_columns:
            example[col] = torch.tensor(example[col]).unsqueeze(0)

        for col in self.dataset.target_columns:
            example[col] = torch.tensor(example[col])

        model_input = {k: v for (k, v) in example.items() if k in self.dataset.input_columns + self.dataset.target_columns}
        # [example[key] = torch.tensor() for ]
        if self.filter_paddings:
            input_len = example['attention_mask'].sum().item()

            for input_key in self.dataset.input_columns:
                model_input[input_key] = model_input[input_key][:, :input_len]

        output = self.model(**model_input, output_attentions=True)
        logits = output['logits']
        input_saliency = compute_input_saliency(self.model, len(example['tokens']), logits)
        output['loss'].backward(retain_graph=True)
        results['loss'] = output['loss'].item()
        results['input_saliency'] = input_saliency
        results['output'] = output['logits'].squeeze(0).tolist()
        results['attn'] = format_attention(output['attentions'], self.num_attention_heads, self.pruned_heads)
        results['attn_pattern'] = format_attention_image(np.array(results['attn']))
        results['head_importance'] = normalize(get_taylor_importance(self.model)).tolist()
        results['tokens'] = example['tokens'][:input_len] if self.filter_paddings else example['tokens']

        return results

# redirect requests from root to index.html
@app.route('/')
def index():
    return flask.render_template(
        'index.html',
        headings=t3_vis.table_headings,
        content=t3_vis.table_content,
        checkpoints=t3_vis.checkpoint_dirs,
        num_hidden_layers=range(t3_vis.num_hidden_layers + 1),
    )


def check_resource_dir(resource_dir):
    
    # Check all subdirectories in the "resource_dir" for data files needed for visualization
    sub_dirs = os.listdir(resource_dir)

    required_files = [
        # (File Name, File Information, Required)
        ('aggregate_attn.pt', 'Aggregated Attention Matrices', True),
        ('head_importance.pt', 'Head Importance Scores', True),
        ('projection_data.pt', 'Dataset Projection Data', True), 
        ('model.pt', 'Model Parameters', False)] # By default use the randomly initialized model parameters

 
    for subdir in sub_dirs:
        subdir_path = pjoin(resource_dir, subdir)
        if os.path.isdir(subdir_path):
            subdir_files = set(os.listdir(subdir_path))

            for (filename, info, required) in required_files:
                if not (filename in subdir_files):
                    print(f"Cannot find file \"{filename}\" containing \"{info}\" in subdirectory \"{subdir_path}\"")
                    if required:
                        return False

    return True




@app.route('/api/data', methods=['POST'])
def get_data():
    """
    Function for retrieving and transforming data for visualization from "resource_dir"

    TODO: Move this function into a method for "T3-Vis"
    """
    projection_type = flask.request.json['projectionType']
    # pdb.set_trace()
    if projection_type == 'cartography' and flask.request.json['checkpointName'] in ['pretrained', 'epoch_1']:
        flask.request.json['checkpointName'] = 'epoch_2'

    t3_vis.curr_checkpoint_dir = flask.request.json['checkpointName']
    checkpoint_dir = pjoin(t3_vis.resource_dir, t3_vis.curr_checkpoint_dir)
    print(checkpoint_dir)
    model_weights_file = pjoin(checkpoint_dir, 'model.pt')
    if os.path.exists(model_weights_file):
        t3_vis.model.load_state_dict(torch.load(model_weights_file))

    layer = int(flask.request.json['hiddenLayer'])

    # TODO: This should be formatted by the user during preprocessing
    projection_keys = {
        'hidden': (f'layer_{layer}_tsne_1', f'layer_{layer}_tsne_2'),
        # 'hidden': (f'projection_{layer}_1', f'projection_{layer}_2'),
        'cartography': ('avg_variability', 'avg_confidence'),
        'discrete': ['labels', 'predictions'],
        'continuous': ['gt_confidence', 'loss'],
    }

    projection_data = torch.load(pjoin(checkpoint_dir, 'projection_data.pt'))
    results = {}
    results['ids'] = projection_data['id'].tolist()

    x_name = projection_keys[projection_type][0]
    y_name = projection_keys[projection_type][1]

    if (x_name in projection_data.keys() and y_name in projection_data.keys()):
        results['x'] = projection_data[x_name].tolist()
        results['y'] = projection_data[y_name].tolist()

        # Avoid scaling t-SNE embeddings
        # TODO: need another option to determine when or when not to scale
        results['domain'] = (min(min(results['x']), min(results['y'])), max(max(results['x']), max(results['y'])))

    results['discrete'] = []
    results['continuous'] = []

    # Process discrete data attributes
    non_discrete_types = [np.float32, np.float64, np.float16, np.double]
    for attr_name in projection_keys['discrete']:
        if attr_name not in projection_data.keys():
            continue

        attr_val = {}
        attr_val['name'] = attr_name

        if projection_data[attr_name].dtype in non_discrete_types:
            projection_data[attr_name] = projection_data[attr_name].astype(int)
        
        attr_val['values'] = projection_data[attr_name].tolist()
        attr_val['domain'] = projection_data[attr_name].astype(str).unique().tolist()
        results['discrete'].append(attr_val)

    # Process continuous data attributes
    for attr_name in projection_keys['continuous']:
        if attr_name not in projection_data.keys():
            continue

        attr_val = {}
        attr_val['name'] = attr_name
        attr_val['values'] = projection_data[attr_name].tolist()
        attr_val['max'] = projection_data[attr_name].max()
        attr_val['min'] = projection_data[attr_name].min()
        attr_val['mean'] = projection_data[attr_name].mean()
        attr_val['median'] = projection_data[attr_name].median()
        results['continuous'].append(attr_val)


    aggregate_attn = torch.load(pjoin(checkpoint_dir, 'aggregate_attn.pt'))

    # Do this for now, need to send an image file to be more efficient
    for i in range(len(aggregate_attn)):
        aggregate_attn[i]['attn'] = json.dumps(aggregate_attn[i]['attn'])

    importance = torch.load(pjoin(checkpoint_dir, 'head_importance.pt'))
    results['head_importance'] = importance.tolist()
    results['aggregate_attn'] = aggregate_attn

    return flask.jsonify(results)


@app.route('/api/eval_one', methods=['POST'])
def eval_one():
    """
    Evaluate a single example in the back-end, specified by the example_id
    """
    sample_name = flask.request.json['example_id']
    # pdb.set_trace()
    results = {}
    results['attn'] = {}

    heads_to_prune = {int(k): v for k, v in flask.request.json['pruned_heads'].items()}

    if heads_to_prune != {}: 
        t3_vis.prune_heads(heads_to_prune)

    results = t3_vis.evaluate_example(int(sample_name))
    return flask.jsonify(results)


if __name__ == '__main__':

    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=True)
    parser.add_argument("--port", default="8888")
    parser.add_argument("--host", default=None)

    # TODO: Let user select the model/dataset from UI interaction
    parser.add_argument("--model", required=True, help="Method for returning the model")
    parser.add_argument("--dataset", required=True, help="Method for returning the dataset")

    # This should be based on the number of examples saved
    parser.add_argument("--n_examples", default=5000, type=int, help="The maximum number of data examples to visualize")

    parser.add_argument("--filter_paddings", default=True, type=bool, help="Filter padding tokens for visualization")
    parser.add_argument("--resource_dir", default=pjoin(cwd, 'resources'), \
                        help="Directory containing the necessary visualization resources for each model checkpoint")


    

    args = parser.parse_args()

    if not check_resource_dir(args.resource_dir):
        exit(1)

    global t3_vis

    t3_vis = T3_Visualization(args)

    app.run(host=args.host, port=int(args.port), debug=args.debug)
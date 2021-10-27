import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd

from utils.head_importance import compute_importance
from utils.helpers import normalize, output_hidden, compute_aggregated_attn

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm
import pdb
import math
from os.path import join as pjoin



def main(args):
    """
    
    """

    try:
        exec(f"from dataset import {args.dataset}")
    except ImportError:
        print(f"\nWarning: Cannot import function \"{args.dataset}\" from directory \"dataset\", please ensure the function is defined in this file!")
    dataset = eval(f"{args.dataset}()")
    columns = dataset.input_columns + dataset.target_columns
    dataset.set_format(type='torch', columns=columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    try:
        exec(f"from models import {args.model}")
    except ImportError:
        print(f"\nWarning: Cannot import function \"{args.model}\" from directory \"models\", please ensure the function is defined in this file!")
    model = eval(f"{args.model}()")


    if not os.path.exists(args.resource_dir):
        os.makedirs(args.resource_dir, exist_ok=True)


    if torch.cuda.is_available():
        model = model.cuda()

    # File names to be saved
    head_importance_file = 'head_importance.pt'
    aggregate_attn_file = 'aggregate_attn.pt'
    projection_data_file = 'projection_data.pt'

    # Pre-process data for pretrained (un-finetuned) model
    pretrained_dir = pjoin(args.resource_dir, 'pretrained')
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir, exist_ok=True)

    head_importance_path = pjoin(pretrained_dir, head_importance_file)
    if not os.path.isfile(head_importance_path):
        importance = compute_importance(model, dataloader)
        importance = normalize(importance)
        torch.save(importance, head_importance_path)

    aggregate_attn_path = pjoin(pretrained_dir, aggregate_attn_file)
    if not os.path.isfile(aggregate_attn_file):
        attn = compute_aggregated_attn(model, dataloader, dataset.max_length)
        torch.save(attn, aggregate_attn_path)

    dataset.set_format(type='torch', columns=columns + ['idx'])
    projection_data_path = pjoin(pretrained_dir, projection_data_file)
    if not os.path.isfile(projection_data_path):
        tsne_hidden, labels = output_hidden(model, dataloader, max_entries=args.n_examples)
        projection_data = {}
        n_examples = len(labels)
        projection_data['id'] = pd.Series(np.arange(n_examples))
        for layer_idx in range(tsne_hidden.shape[1]):
            projection_data[f'projection_{layer_idx}_1'] = pd.Series(tsne_hidden[:, layer_idx, 0])
            projection_data[f'projection_{layer_idx}_2'] = pd.Series(tsne_hidden[:, layer_idx, 1])

        projection_data['labels'] = pd.Series(labels)
        torch.save(projection_data, projection_data_path)



    # Initializing the optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay":  0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    num_update_steps_per_epoch = len(dataloader) // args.batch_size
    max_steps = math.ceil(args.epochs * num_update_steps_per_epoch)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max_steps)




    # For computing confidence and variance during training
    p_y = np.zeros((len(dataset), args.epochs))
    ids = np.arange(len(dataset))

    for epoch in range(args.epochs):


        # Makedir for current epoch
        epoch_dir = pjoin(args.resource_dir, f"epoch_{epoch + 1}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir, exist_ok=True)

        epoch_iterator = dataloader
        model.train()

        projection_data = {}

        # Checkpoint and data path for current epoch
        ckpt_path = os.path.join(epoch_dir, f"model.pth")

        head_importance_path = pjoin(epoch_dir, head_importance_file)
        aggregate_attn_path = pjoin(epoch_dir, aggregate_attn_file)
        projection_data_path = pjoin(epoch_dir, projection_data_file)

        # Used as keys in projection_data
        labels = np.zeros(len(dataset))
        epoch_loss = np.zeros(len(dataset))
        predictions = np.zeros(len(dataset))

        # Iterating through all optimization steps (mini-batches)
        for step, inputs in enumerate(tqdm(epoch_iterator)):

            batch_size_ = inputs['input_ids'].__len__()

            if torch.cuda.is_available():
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.cuda()
            idx = inputs['idx'].cpu().tolist()
            del inputs['idx']

            labels_ = inputs['labels'].cpu().numpy()
            labels[idx] = inputs['labels'].cpu().numpy()

            output = model(**inputs)
            batch_loss = output['loss']
            batch_loss.backward()


            logits = output['logits']
            probs = torch.nn.functional.softmax(logits, dim=1)

            predictions[idx] = probs.max(axis=1)[1].cpu().numpy()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_sample = loss_fct(logits, inputs['labels']).detach().cpu().numpy()

            p_y_ = np.zeros(batch_size_)
            for i in range(batch_size_):
                p_y_[i] = probs[i][labels_[i]].cpu().item()


            for i in range(batch_size_):
                epoch_loss[idx] = loss_per_sample

            p_y[idx, epoch] = p_y_


            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()

        # Save model parameters
        torch.save(model.state_dict(), ckpt_path)

        # Save the necessary data file for the current epoch
        if not os.path.isfile(head_importance_path):
            importance = compute_importance(model, dataloader)
            importance = normalize(importance)
            torch.save(importance, head_importance_path)

        if not os.path.isfile(aggregate_attn_file):
            attn = compute_aggregated_attn(model, dataloader, dataset.max_length)
            torch.save(attn, aggregate_attn_path)

        dataset.set_format(type='torch', columns=columns + ['idx'])
        if not os.path.isfile(projection_data_path):
            tsne_hidden, labels = output_hidden(model, dataloader, max_entries=args.n_examples)
            projection_data = {}
            n_examples = len(labels)
            projection_data['id'] = pd.Series(np.arange(n_examples))
            for layer_idx in range(tsne_hidden.shape[1]):
                projection_data[f'projection_{layer_idx}_1'] = pd.Series(tsne_hidden[:, layer_idx, 0])
                projection_data[f'projection_{layer_idx}_2'] = pd.Series(tsne_hidden[:, layer_idx, 1])

            projection_data['labels'] = pd.Series(labels)
            projection_data['loss'] = pd.Series(epoch_loss.squeeze()[:max_entries])
            projection_data['predictions'] = pd.Series(predictions.squeeze()[:max_entries])
            projection_data['gt_confidence'] = pd.Series(p_y[:max_entries, epoch])
            torch.save(projection_data, projection_data_path)

            if epoch >= 1:
                confidence = p_y[:, :epoch + 1].mean(axis=1)
                variability = ((p_y[:, :epoch + 1] - np.repeat(confidence[:, np.newaxis], epoch + 1, axis=1)) ** 2).mean(axis=1) ** (1/2)
                projection_data['avg_confidence'] = pd.Series(confidence[:max_entries])
                projection_data['avg_variability'] = pd.Series(variability[:max_entries])




# Produce output needed for the demo on SST corpus
if __name__ == "__main__":

    torch.manual_seed(0)
    random.seed(0)

    cwd = os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Method for returning the model")
    parser.add_argument("--dataset", required=True, help="Method for returning the dataset")
    parser.add_argument("--n_examples", default=5000, help="The maximum number of data examples to visualize")

    parser.add_argument("--resource_dir", default=pjoin(cwd, 'resources'), \
                        help="Directory containing the necessary visualization resources for each model checkpoint")

    
    # Hyperparameters for training
    parser.add_argument("--lr", default=2e-5)
    parser.add_argument("--epochs", default=3)
    parser.add_argument("--batch_size", default=16)

    args = parser.parse_args()
    main(args)

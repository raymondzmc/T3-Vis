import torch

from models import 

def train(args):
    torch.backends.cudnn.deterministic = True

    args.dataset_type = 'train'
    train_loader = get_dataloader(args, 'train')
    val_loader = get_dataloader(args, 'valid')
    checkpoint_dir = os.path.join(args.result_dir, 'checkpoints')

    model = get_model(args)
    model.train()

    optim = build_optim(args, model)

    if args.resume_checkpoint != None:
        try:
            model.load_state_dict(torch.load(args.resume_checkpoint, map_location=args.device)['model'])
        except RuntimeError:
            state_dict = torch.load(args.resume_checkpoint, map_location=args.device)['model']
            new_state_dict = collections.OrderedDict()
            for key, item in state_dict.items():
                module_names = key.split('.')
                if module_names[0] == 'bert':
                    module_names[0] = 'encoder' 
                new_key = '.'.join(module_names)
                new_state_dict[new_key] = item
            model.load_state_dict(new_state_dict)


    trainer = Trainer(args,
                      model,
                      train_loader,
                      val_loader,
                      optim)


if __name__ == "__main__":
    parser.add_argument("--model", required=True, help="Method for returning the model")
    parser.add_argument("--dataset", required=True, help="Method for returning the dataset")
    parser.add_argument("--resource_dir", default=pjoin(cwd, 'resources'), \
                        help="Directory containing the necessary visualization resources for each model checkpoint")



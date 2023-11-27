from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel
import torch
import numpy as np
from PaiNN.active_learning import GeneralActiveLearning
import os 
import math
import glob
import json
import argparse, toml
from pathlib import Path
from ase.io import read, write, Trajectory

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Set if you do NEB or MD",
    )
    parser.add_argument(
        "--neb_init",
        type=bool,
        default= True,
        help="if True: Choose all the inital NEB images(not MD) to be labeled including initial and final image. Do not work if you have set dataset as a param",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="How to get features",
    )
    parser.add_argument(
        "--selection",
        type=str,
        help="Selection method, one of `max_dist_greedy`, `deterministic_CUR`, `lcmd_greedy`, `max_det_greedy` or `max_diag`",
    )
    parser.add_argument(
        "--n_random_features",
        type=int,
        help="If `n_random_features = 0`, do not use random projections.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="How many data points should be selected",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Where to find the models",
    )
    parser.add_argument(
        "--dataset", type=str, help="Path to ASE trajectory",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--pool_set", type=str, help="Path to MD trajectory obtained from machine learning potential",
    )
    parser.add_argument(
        "--train_set", type=str, help="Path to training set. Useful for pool/train based selection method",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        if not ns.__dict__.get(k):
            ns.__dict__[k] = v

def main():
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)
    
    setup_seed(args.random_seed)

    # Load models
    model_pth = Path(args.load_model).rglob('*best_model.pth')
    models = []
    for each in model_pth:
        state_dict = torch.load(each) 
        model = PainnModel(
            num_interactions=state_dict["num_layer"], 
            hidden_state_size=state_dict["node_size"], 
            cutoff=state_dict["cutoff"],
        )
        model.to(args.device)
        model.load_state_dict(state_dict["model"])    
        models.append(model)
        
    # Load dataset
    if args.dataset:
        with open(args.split_file, 'r') as f:
            datasplits = json.load(f)
            
        dataset = AseDataset(args.dataset, cutoff=models[0].cutoff)
        data_dict = {
            'pool': torch.utils.data.Subset(dataset, datasplits['pool']),
            'train': torch.utils.data.Subset(dataset, datasplits['train']),
        }
    elif args.pool_set and args.train_set:
        if isinstance(args.pool_set, list):
            dataset = []
            for traj in args.pool_set:
                if Path(traj).stat().st_size > 0:
                    dataset += read(traj, index=':') 
        else:
            if args.neb_init and args.method == 'NEB': # If we want to label the first N images in NEB
                # Load the number of NEB images
                txt_sep = params['pool_set'].split('/')
                neb_path = os.path.join('/',*txt_sep[:-1],'arguments.toml')
                with open(neb_path, 'r') as file:
                    params_neb = toml.load(file)
                neb_img = int(params_neb['num_img']+2)
                dataset = read(args.pool_set,index=f'{neb_img}:')
            else:
                dataset = args.pool_set
        data_dict = {
            'pool': AseDataset(dataset, cutoff=models[0].cutoff),
            'train': AseDataset(args.train_set, cutoff=models[0].cutoff),
        }
    else:
        raise RuntimeError("Please give valid pool data set for selection!")

    # raise error if the pool dataset is not large enough
    if args.method =='MD':
        print(len(data_dict['pool']))
        if len(data_dict['pool']) < args.batch_size * 5: #5
            raise RuntimeError(f"""The pool data set ({len(data_dict['pool'])}) is not large enough for selection!
            It should be larger than 10 times batch size ({args.batch_size*10}).
            Check your MD simulation!""")

    # Select structures
    al = GeneralActiveLearning(
        kernel=args.kernel, 
        selection=args.selection, 
        n_random_features=args.n_random_features,
    )

    # Manually choose N NEB images to be labeled.
    if args.neb_init and args.method == 'NEB':
        indices_neb = np.arange(0,neb_img)
        if neb_img > args.batch_size:
            raise RuntimeError(f"""The pool data set is not large enough for selection! 
            Choose a batch size ({args.batch_size}) larger then NEB selection ({neb_img})!""")
        if neb_img == args.batch_size:
            indices = indices_neb.tolist()
        else:
            indices_al = al.select(models, data_dict, al_batch_size=args.batch_size-neb_img)
            indices = np.concatenate((indices_neb,np.array(indices_al)+neb_img),dtype=int).tolist()
    # Choose N MD images to be labeled.
    elif args.method == 'MD':
        indices = al.select(models, data_dict, al_batch_size=args.batch_size)
    else:
        raise RuntimeError("Please give valid method for selection!")
    al_idx = [datasplits['pool'][i] for i in indices] if args.dataset else indices
    al_info = {
        'kernel': args.kernel,
        'selection': args.selection,
        'dataset': args.dataset if args.dataset else args.pool_set,
        'selected': al_idx,
    }

    with open('selected.json', 'w') as f:
        json.dump(al_info, f)

    # Update new data splits
    if args.dataset:
        pool_idx = np.delete(datasplits['pool'], indices)    
        datasplits['pool'] = pool_idx.tolist()
        datasplits['train'] += al_idx
        with open(args.split_file, 'w') as f:
            json.dump(datasplits, f)

if __name__ == "__main__":
    main()

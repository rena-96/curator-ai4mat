import json, toml, sys
from pathlib import Path
from myqueue.workflow import run
from typing import List, Dict
from ase.io import Trajectory, read, write
import numpy as np
import copy
import os 
# args parsing

with open('config.toml') as f:
    args = toml.load(f)

# get absolute path
name_list = [
    'dataset', 
    'split_file', 
    'load_model', 
    'init_traj', 
    'pool_set', 
    'train_set', 
    'label_set',
    'al_info',
    'root'
]
def get_absolute_path(d: dict):
    for k, v in d.items():
        if k in name_list and not Path(v).is_absolute():
            d[k] = str(Path(v).resolve())
        elif isinstance(v, dict):
            d[k] = get_absolute_path(v)
    return d
args = get_absolute_path(args)

# parsing training parameters
train_params = {}
if args['train'].get('ensemble'):
    for name, params in args['train']['ensemble'].items():
        for k, v in args['train'].items():
            if not isinstance(v, dict) and k not in params:
                params[k] = v
        train_params[name] = params
else:
    params = {}
    for k, v in args['train'].items():
        if not isinstance(v, dict) and k not in params:
            params[k] = v
    train_params['model'] = params
# train_resource = args['train']['resource']

# parsing active learning parameters
al_params = {}
if args['select'].get('runs'):
    for name, params in args['select']['runs'].items():
        for k, v in args['select'].items():
            if not isinstance(v, dict) and k not in params:
                params[k] = v
        al_params[name] = params
else:
    params = {}
    for k, v in args['select'].items():
        if not isinstance(v, dict) and k not in params:
            params[k] = v
    al_params['select'] = params

# al_resource = args['select']['resource']

# parsing MD parameters
md_params = {}
try:
    if args['MD'].get('runs'):
        for name, params in args['MD']['runs'].items():
            for k, v in args['MD'].items():
                if not isinstance(v, dict) and k not in params:
                    params[k] = v
            md_params[name] = params
    else:
        params = {}
        for k, v in args['MD'].items():
            if not isinstance(v, dict) and k not in params:
                params[k] = v
        md_params['md_run'] = params
except:
    pass

# parsing NEB parameters
neb_params = {}
try:
    if args['NEB'].get('runs'):
        for name, params in args['NEB']['runs'].items():
            for k, v in args['NEB'].items():
                if not isinstance(v, dict) and k not in params:
                    params[k] = v
            neb_params[name] = params
    else:
        params = {}
        for k, v in args['NEB'].items():
            if not isinstance(v, dict) and k not in params:
                params[k] = v
        neb_params['neb_run'] = params
except:
    pass

# DFT labelling
dft_params = {}
tmp_params = {k: v for k, v in args['labeling'].items() if not isinstance(v, dict)}

with open('labeling_config.toml') as f:
    args_labeling = toml.load(f)

tmp_params['VASP'] = args_labeling['VASP']
tmp_params['GPAW'] = args_labeling['GPAW']
if args['labeling'].get('runs'):
    for name, params in args['labeling']['runs'].items():
        new_params = copy.deepcopy(tmp_params)
        if type(params) is not dict:
            tmp_params[name] = params
            continue
        
        for k, v in params.items():
            # Update VASP or GPAW parameters to local parameters. Note that they will only be included if the local parameters are presented in VASP/GPAW parameters.
            if k in new_params['VASP']:
                new_params['VASP'][k] = v
            elif k in new_params['GPAW']:
                new_params['GPAW'][k] = v
            else:
                new_params[k] = v # if there is no VASP/GPAW parameters to update
        dft_params[name] = new_params
else:
    dft_params['dft_run'] = tmp_params


root = args['global']['root']

def train_models(folder, deps, extra_args: List[str] = [], iteration: int=0):
    tasks = []
    node_info = args['train']['resource']
    # parse parameters 
    for name, params in train_params.items():
        path = Path(f'{folder}/iter_{iteration}/{name}')

        if not params.get('start_iteration'):
            params['start_iteration'] = 0
        if iteration >= params['start_iteration']:
            if not path.is_dir():
                path.mkdir(parents=True)

            # load model
            if iteration > 0:
                load_model = f'{root}/{folder}/iter_{iteration-1}/{name}/{params["output_dir"]}/best_model.pth'
                if Path(load_model).is_file():
                    params['load_model'] = load_model
    #         elif iteration == 0:
    #             params['load_model'] = f'/home/scratch3/xinyang/Au-facets/old_training/train/{name}/model_output/best_model.pth'

            with open(path / 'arguments.toml', 'w') as f:
                toml.dump(params, f)

            arguments = ['--cfg', 'arguments.toml']                
            arguments += extra_args

            tasks.append(run(
                script=f'{root}/train.py', 
                nodename='' if not node_info.get('nodename') else node_info['nodename'],
                #cores=8 if not node_info.get('cores') else node_info['cores'], 
                tmax='2h' if not node_info.get('tmax') else node_info['tmax'],
                args=arguments,
                folder=path,
                name=name,
                deps=deps,
            ))
        
    return tasks

def active_learning(folder, deps, extra_args: List[str] = [], iteration: int=0):
    tasks = {}
    node_info = args['select']['resource']
    # parse parameters 
    for name, params in al_params.items():
        path = Path(f'{folder}/iter_{iteration}/{name}')
        if not params.get('start_iteration'):
            params['start_iteration'] = 0
        if iteration >= params['start_iteration']:
            if not path.is_dir():
                path.mkdir(parents=True)
            
            params['load_model'] = f'{root}/train/iter_{iteration}'
            if not params.get('dataset'):
                if params.get('method')=='MD':
                    params['pool_set'] = [f'{root}/md/iter_{iteration}/{name}/MD.traj', f'{root}/md/iter_{iteration}/{name}/warning_struct.traj']
                elif params.get('method')=='NEB':
                    params['pool_set'] = f'{root}/neb/iter_{iteration}/{name}/NEB_MD.traj'
                else:
                    raise RuntimeError("Please give valid method for selection!")

            with open(path / 'arguments.toml', 'w') as f:
                toml.dump(params, f)

            arguments = ['--cfg', 'arguments.toml']
            
            # Choose the dependence, where deps=[md,neb]
            
            if type(deps) == list:
                if params.get('method')=='MD':
                    tasks[name] = run(
                    script=f'{root}/al_select.py', 
                    nodename='' if not node_info.get('nodename') else node_info['nodename'],
                    cores=8 if not node_info.get('cores') else node_info['cores'], 
                    tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
                    args=arguments,
                    folder=path,
                    name=name,
                    deps=[deps[0][name]],
                    )
                elif params.get('method')=='NEB':
                    tasks[name] = run(
                    script=f'{root}/al_select.py', 
                    nodename='' if not node_info.get('nodename') else node_info['nodename'],
                    cores=8 if not node_info.get('cores') else node_info['cores'], 
                    tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
                    args=arguments,
                    folder=path,
                    name=name,
                    deps=[deps[1][name]],
                )
            else:
                tasks[name] = run(
                    script=f'{root}/al_select.py', 
                    nodename='' if not node_info.get('nodename') else node_info['nodename'],
                    cores=8 if not node_info.get('cores') else node_info['cores'], 
                    tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
                    args=arguments,
                    folder=path,
                    name=name,
                    deps=[deps[name]],
                )
    
    return tasks

def run_md(folder, deps=[], extra_args: List[str] = [], iteration: int=0):
    tasks = {}
    node_info = args['MD']['resource']
    for name, params in md_params.items():
        path = Path(f'{folder}/iter_{iteration}/{name}')

        if not params.get('start_iteration'):
            params['start_iteration'] = 0
        if iteration >= params['start_iteration']:
            if not path.is_dir():
                path.mkdir(parents=True)
            params['load_model'] = f'{root}/train/iter_{iteration}'

                
            if iteration > params['start_iteration']:
                params['init_traj'] = f'{root}/md/iter_{iteration-1}/{name}/MD.traj'

            with open(path / 'arguments.toml', 'w') as f:
                toml.dump(params, f)

            arguments = ['--cfg', 'arguments.toml']
            
            tasks[name] = run(
                script=f'{root}/md_run.py', 
                nodename='' if not node_info.get('nodename') else node_info['nodename'],
                cores=8 if not node_info.get('cores') else node_info['cores'], 
                tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
                args=arguments,
                folder=path,
                name=name,
                deps=deps,
            )

    return tasks

def run_neb(folder, deps=[], extra_args: List[str] = [], iteration: int=0):
    tasks = {}
    node_info = args['NEB']['resource']
    for name, params in neb_params.items():
        path = Path(f'{folder}/iter_{iteration}/{name}')

        if not params.get('start_iteration'):
            params['start_iteration'] = 0
        if iteration >= params['start_iteration']:
            if not path.is_dir():
                path.mkdir(parents=True)
            params['load_model'] = f'{root}/train/iter_{iteration}'

                
            #if iteration > params['start_iteration']:
            #    params['init_traj'] = f'{root}/neb/iter_{iteration-1}/{name}/NEB.traj'

            with open(path / 'arguments.toml', 'w') as f:
                toml.dump(params, f)

            arguments = ['--cfg', 'arguments.toml']
            
            tasks[name] = run(
                script=f'{root}/neb_run.py', 
                nodename='' if not node_info.get('nodename') else node_info['nodename'],
                cores=8 if not node_info.get('cores') else node_info['cores'], 
                tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
                args=arguments,
                folder=path,
                name=name,
                deps=deps,
            )

    return tasks

def run_dft(folder, deps={}, extra_args: List[str] = [], iteration: int=0):
    tasks = []
    #os.system('conda init bash ')
    print('Here')
    #os.system('conda activate gpaw ')
    node_info = args['labeling']['resource']
    for name, params in dft_params.items():
        path = Path(f'{folder}/iter_{iteration}/{name}')
        if not params.get('start_iteration'):
            params['start_iteration'] = 0
        if iteration >= params['start_iteration']:
            if not path.is_dir():
                path.mkdir(parents=True)

            # get images that need to be labeled
            params['system'] = name
            params['al_info'] = f'{root}/select/iter_{iteration}/{name}/selected.json'
            with open(path / 'arguments.toml', 'w') as f:
                toml.dump(params, f)

            arguments = ['--cfg', 'arguments.toml']
            
            if params.get('num_jobs'):
                for i in range(params['num_jobs']):
                    dft_arguments = ['--cfg', '../arguments.toml', '--job_order', f'{i}']
                    dft_path = path / f'{i}'
                    if not dft_path.is_dir():
                        dft_path.mkdir(parents=True)
                    tasks.append(run(
                        script=f'{root}/labeling.py', 
                        nodename='' if not node_info.get('nodename') else node_info['nodename'],
                        cores=40 if not node_info.get('cores') else node_info['cores'], 
                        tmax='50h' if not node_info.get('tmax') else node_info['tmax'],
                        args=dft_arguments,
                        folder=dft_path,
                        name=name,
                        deps=[deps[name]],
                    ))
            else:
                tasks.append(run(
                    script=f'{root}/labeling.py', 
                    nodename='' if not node_info.get('nodename') else node_info['nodename'],
                    cores=40 if not node_info.get('cores') else node_info['cores'], 
                    tmax='50h' if not node_info.get('tmax') else node_info['tmax'],
                    args=arguments,
                    folder=path,
                    name=name,
                    deps=[deps[name]],
                ))

    return tasks

def all_done(runs):
    return all([task.done for task in runs])

def workflow():
    dft = []
    for iteration in range(2):
        # training part
        training = train_models('train', deps=dft, iteration=iteration)

        if len(md_params)!=0 and len(neb_params)!=0:
            print('Here')
            # data generating for both md and neb
            md = run_md('md', deps=training, iteration=iteration)
            neb = run_neb('neb', deps=training, iteration=iteration)
            # active learning selection
            select = active_learning('select', deps=[md,neb], iteration=iteration)
        elif len(md_params)!=0 and len(neb_params)==0:
            # data generating for only MD
            md = run_md('md', deps=training, iteration=iteration)
            # active learning selection
            select = active_learning('select', deps=md, iteration=iteration)
        elif len(neb_params)!=0 and len(md_params)==0:
            # data generating for only NEB
            neb = run_neb('neb', deps=training, iteration=iteration)
            # active learning selection
            select = active_learning('select', deps=neb, iteration=iteration)
        # DFT labeling
        dft = run_dft('labeling', deps=select, iteration=iteration)

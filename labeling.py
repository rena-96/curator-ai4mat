#from ase.calculators.vasp import Vasp
from ase.io import read, write, Trajectory
from ase.db import connect
from shutil import copy
import os, subprocess
import numpy as np
import argparse
import json
import toml
from pathlib import Path
from ase.calculators.calculator import CalculationFailed
for k, v in os.environ.items():
    if k.startswith('SLURM'):
        print(k, v)
#print(os.environ['SLURM_SUBMIT_DIR'])
from gpaw import GPAW, KohnShamConvergenceError

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--label_set",
        type=str,
        help="Path to trajectory to be labeled by DFT",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        help="Path to existing training data set",
    )
    parser.add_argument(
        "--pool_set", 
        type=str, 
        help="Path to MD trajectory obtained from machine learning potential",
    )
    parser.add_argument(
        "--al_info", 
        type=str, 
        help="Path to json file that stores indices selected in pool set",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        help="Number of DFT jobs",
    )
    parser.add_argument(
        "--job_order",
        type=int,
        help="Split DFT jobs to several different parts",
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
        if not isinstance(v, dict):
            ns.__dict__[k] = v

def main():
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        # Set the data set selected by the active learning 
        with open(params['al_info']) as f:
            params['pool_set'] = json.load(f)['dataset'] #[f'{root}/md/iter_{iteration}/{name}/MD.traj', f'{root}/md/iter_{iteration}/{name}/warning_struct.traj']            
        update_namespace(args, params)

    # Set up dataframe and load possible converged data id's
    db = connect('dft_structures.db')
    db_al_ind = [row.al_ind for row in db.select([('converged','=','True')])] #
    # get images and set parameters
    if args.label_set:
        images = read(args.label_set, index = ':')
    elif args.pool_set:
        if isinstance(args.pool_set, list):
            pool_traj = []
            for pool_path  in args.pool_set:
                if Path(pool_path).stat().st_size > 0:
                    pool_traj += read(pool_path, ':')
        else:
            pool_traj = Trajectory(args.pool_set)
        
        with open(args.al_info) as f:
            indices = json.load(f)["selected"]
        
        if db_al_ind:
            _,rm,_ = np.intersect1d(indices, db_al_ind,return_indices=True)
            indices = np.delete(indices,rm)
        if args.num_jobs:
            split_idx = np.array_split(indices, args.num_jobs)
            indices = split_idx[args.job_order]
        images = [pool_traj[i] for i in indices]        
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')
    
    vasp_params = params['VASP']
    gpaw_params = params['GPAW']
    check_result = False

    if params['method'] =='VASP':
        # set environment variables
        os.putenv('ASE_VASP_VDW', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        os.putenv('VASP_PP_PATH', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        os.putenv('ASE_VASP_COMMAND', 'mpirun vasp_std')
        calc = Vasp(**vasp_params)
        unconverged_idx = []
        for i, atoms in enumerate(images):
            al_ind = indices[i]
            atoms.set_pbc([True,True,True])
            atoms.set_calculator(calc)
            try:
                atoms.get_potential_energy()
            except CalculationFailed:
                check_result = True
                db.write(atoms,al_ind=al_ind,converged=False)
                unconverged_idx.append(i)
                copy('OSZICAR', f'OSZICAR_{i}_{al_ind}')
                continue

            steps = int(subprocess.getoutput('grep LOOP OUTCAR | wc -l'))
            if steps <= vasp_params['nelm']:
                db.write(atoms,al_ind=al_ind,converged=True)
            else:
                check_result = True
                db.write(atoms,al_ind=al_ind,converged=False)
                unconverged_idx.append(i)
            copy('OSZICAR', f'OSZICAR_{i}_{al_ind}')
            os.remove('WAVECAR')
            os.remove('CHGCAR')

    elif params['method'] =='GPAW':
	    
        calc = GPAW(**gpaw_params)
        calc.set(txt='GPAW.txt')
        unconverged_idx = []
        for i, atoms in enumerate(images):
            al_ind = indices[i]
            atoms.set_pbc([True,True,True])
            atoms.set_calculator(calc)
            try:
                atoms.get_potential_energy()
            except KohnShamConvergenceError:
                check_result = True
                db.write(atoms,al_ind=al_ind,converged=False)
                unconverged_idx.append(i)
                copy('GPAW.txt', f'GPAW_{i}_{al_ind}.txt')
                continue

            db.write(atoms,al_ind=al_ind,converged=True)
            copy('GPAW.txt', f'GPAW_{i}_{al_ind}.txt')
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')

    
    if check_result:
        raise RuntimeError(f"DFT calculations of {unconverged_idx} are not converged!")
    # write to training set
    if args.train_set:
        train_traj = Trajectory(args.train_set, mode = 'a')
        database = connect('dft_structures.db')#read('dft_structures.traj', ':')
        for row in database.select([('converged','=','True')]):
            atom = row.toatoms()
            atom.info['system'] = args.system
            atom.info['path'] = str(Path('dft_structures.db').resolve())
            train_traj.write(atom)

if __name__ == "__main__":
    main()

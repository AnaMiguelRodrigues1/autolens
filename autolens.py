import argparse
import subprocess
from subprocess import DEVNULL, STDOUT
import os
import venv
import sys

def run_external_command(command):
    #subprocess.check_call(command, stdout=DEVNULL, stderr=DEVNULL, shell=True)
    subprocess.check_call(command, shell=True)

def maybe_prepare_requirements(mode, cache_dir):
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_dir = os.path.abspath(cache_dir)
    
    venv_name = f"venv-{mode}"
    venv_path = os.path.join(cache_dir, venv_name)
    
    # prepare a venv for the specific package if its not available
    if not os.path.exists(venv_path):
        run_external_command(f"{sys.executable} -m venv {venv_path}")
        #venv.create(venv_path)
        pip_command = os.path.join(venv_path,"bin","pip")
        run_external_command(f"{pip_command} install --upgrade pip")
        
        run_external_command(f"{pip_command} install -r requirements_{mode}.txt")
        
    # add venv packages to the path
    sys.path.insert(1, os.path.join(venv_path,"bin","python"))
    #return os.path.join(venv_path,"bin","python")

def autolens(autolens_mode, 
             dataset_path, 
             target_size,
             test_percentage,
             val_percentage,
             clean_metadata,
             cache_dir):
    
    maybe_prepare_requirements(args.autolens_mode, args.cache_dir)
    
    # REST OF THE CODE TO RUN
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Example script.')

    parser.add_argument('autolens_mode', type=str, help='TODO ana') # options
    parser.add_argument('dataset_path', type=str, help='dataset path TODO ana')
    # ADD abstraction 
    parser.add_argument('--target_size', type=tuple, default=(255,255), help='dataset size')
    parser.add_argument('--test_percentage', type=float, default=0.2, help='dataset size')
    parser.add_argument('--val_percentage', type=float, default=0.1, help='dataset size')
    parser.add_argument('--clean_metadata', action="store_true", help='dataset size')
    parser.add_argument('--cache_dir', type=str, default="~/.cache/autolens" ,help='dataset size')

    # Parse the arguments
    args = parser.parse_args()
    
    autolens(args.autolens_mode, 
             args.dataset_path,
             args.target_size,
             args.test_percentage,
             args.val_percentage,
             args.clean_metadata,
             args.cache_dir)
    
    print(sys.path)
    
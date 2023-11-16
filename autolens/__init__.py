__version__="0.1.0"

import argparse
import subprocess
from subprocess import DEVNULL, STDOUT
import os
import venv
import sys
import importlib
import shutil

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
    #print(venv_path)
    #print(sys.executable.split("/")[-1])
    #print(os.path.join(venv_path,"lib64",sys.executable.split("/")[-1],"site-packages"))
    sys.path.insert(1, os.path.join(venv_path,"lib64",sys.executable.split("/")[-1],"site-packages"))
    #return os.path.join(venv_path,"bin","python")

def autolens(autolens_mode, 
             dataset_path, 
             target_size,
             test_percentage,
             val_percentage,
             clean_metadata,
             cache_dir):
    
    maybe_prepare_requirements(autolens_mode, cache_dir)
    
    # run ana's super code
    package_name = autolens_mode.upper()
    
    print(sys.path)
    
    module = importlib.import_module(f"autolens.{package_name}.run")
    automl_main = getattr(module, "main")
    
    automl_main(dataset_path,
                steps=1,
                target_size=target_size,
                test_size=test_percentage,
                valid_size=val_percentage)
    
    # delete "metadata" which is not metadata but ana says its metadata
    if clean_metadata:
        _path = f"resources/{autolens_mode}"
        if os.path.exists(_path):
            shutil.rmtree(_path)
        
    

    
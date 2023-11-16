from autolens import autolens
import os
import argparse

if __name__ == "__main__":
    
    home_dir = os.path.expanduser("~")
    
    parser = argparse.ArgumentParser(description='Example script.')

    parser.add_argument('autolens_mode', type=str, help='TODO ana', choices=["ludwig", "fastai"]) # options
    parser.add_argument('dataset_path', type=str, help='dataset path TODO ana')
    # ADD abstraction 
    parser.add_argument('--target_size', type=tuple, default=(255,255), help='dataset size')
    parser.add_argument('--test_percentage', type=float, default=0.2, help='dataset size')
    parser.add_argument('--val_percentage', type=float, default=0.1, help='dataset size')
    parser.add_argument('--clean_metadata', action="store_true", help='dataset size')
    parser.add_argument('--cache_dir', type=str, default=f"{home_dir}/.cache/autolens" ,help='dataset size')

    # Parse the arguments
    args = parser.parse_args()
    
    autolens(args.autolens_mode, 
             args.dataset_path,
             args.target_size,
             args.test_percentage,
             args.val_percentage,
             args.clean_metadata,
             args.cache_dir)
    
    #print(sys.path)
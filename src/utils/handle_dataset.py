from src.utils import handle_hist, handle_pneum, handle_brain

datasets = {'../../BreaKHis_v1/': handle_hist.create_dataset, 
        '../../chest_xray/': handle_pneum.create_dataset,
        '../../brain_mri/': handle_brain.create_dataset
        }

def check(path_dataset):
    for key,value in datasets.items():
        if path_dataset == key:
            dataset = value(path_dataset)
            
            return dataset

    return None

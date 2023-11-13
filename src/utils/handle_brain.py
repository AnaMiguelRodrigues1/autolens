import os
import pandas as pd

from src.dataset.build import Dataset

def create_dataset(path_dataset: str, target_class: str='meningioma_tumor', binary=True, test_size=0.2, valid_size=0.1) -> Dataset:
    """
    :param path_dataset:
    :return:
    """

    df = data('../../brain_mri/', target_class)

    if binary:
        df.drop("multiclass_label", axis="columns", inplace=True)
        tag = "binary"
    else:
        df.drop("binary_label", axis="columns", inplace=True)
        tag = "multiclass"

    try:
        os.mkdir("./resources")
    except FileExistsError:
        print("Resources already created")
    
    # this path is now the dataset metadata for training
    df.to_csv(f"./resources/metadata_braintumor_{tag}.csv")

    return Dataset(f"resources/metadata_braintumor_{tag}.csv", path_dataset, df.shape[0], test_size, valid_size)

def load_dataset(path_metadata: str, path_dataset: str) -> Dataset:
    df = pd.read_csv(path_metadata)

    return Dataset(path_metadata, path_dataset, df.shape[0])

def data(path_dataset: str, target_class: str) -> pd.DataFrame:
    paths = ['Testing', 'Training']
    classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
     
    data = {'filename': [], 'binary_label': [], 'multiclass_label': []}
    
    condition_columns = {
        'no_tumor': 0,
        'meningioma_tumor': 1,
        'glioma_tumor': 2,
        'pituitary_tumor': 3
    }

    # Creating Multi-Class Labels
    for c in classes:
        for i in os.listdir(path_dataset + paths[1] + '/' + c):
            data['filename'].append(path_dataset + paths[1] + '/' + c + '/' + i)
            data['multiclass_label'].append(condition_columns[c])
            if c == target_class:
                data['binary_label'].append(1)
            else:
                data['binary_label'].append(0)

    df = pd.DataFrame(data)

    return df

import os
import pandas as pd

from src.dataset.build import Dataset

def create_dataset(path_dataset: str) -> Dataset:
    """
    :param path_dataset:
    :return:
    """

    df = data('../../chest_xray')

    try:
        os.mkdir("./resources")
    except FileExistsError:
        print("Resources already created")

    # this path is now the dataset metadata for training
    df.to_csv(f"./resources/metadata_pneumonia_binary.csv")

    return Dataset(f"resources/metadata_pneumonia_binary.csv", path_dataset, df.shape[0])

def load_dataset(path_metadata: str, path_dataset: str) -> Dataset:
    df = pd.read_csv(path_metadata)

    return Dataset(path_metadata, path_dataset, df.shape[0])

def data(path_dataset: str) -> pd.DataFrame:
    paths = ['/test', '/train', '/val']
    classes = ['/NORMAL', '/PNEUMONIA']

    data = {'filename': [], 'binary_label': []}

    for p in paths:
        for c in classes:
            complete_path = path_dataset + p + c
            if c.endswith('NORMAL'):
                label = 0
            else:
                label = 1
            for filename in os.listdir(complete_path):
                file_path = os.path.join(complete_path, filename)
                data['filename'].append(file_path)
                data['binary_label'].append(label)

    df = pd.DataFrame(data)
    
    return df

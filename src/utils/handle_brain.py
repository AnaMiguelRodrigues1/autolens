import os
import pandas as pd

from src.dataset.build import Dataset

def create_dataset(path_dataset: str, binary=False) -> Dataset:
    """
    :param path_dataset:
    :return:
    """

    df = data('../../brain_mri/')

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

    return Dataset(f"resources/metadata_braintumor_{tag}.csv", path_dataset, df.shape[0])

def load_dataset(path_metadata: str, path_dataset: str) -> Dataset:
    df = pd.read_csv(path_metadata)

    return Dataset(path_metadata, path_dataset, df.shape[0])

def data(path_dataset: str) -> pd.DataFrame:
    paths = ['/Testing', '/Training']
    classes = ['/glioma_tumor', '/meningioma_tumor', '/no_tumor', '/pituitary_tumor']
     
    data = {'filename': [], 'multiclass_label': []}

    # Creating Multi-Class Labels
    for p in paths:
        for c in classes:
            complete_path = path_dataset + p + c
            if c.endswith('no_tumor'):
                label = 0 
            elif c.endswith('glioma_tumor'):
                label = 1 
            elif c.endswith('pituitary_tumor'):
                label = 2
            else:
                label = 3
            for filename in os.listdir(complete_path):
                file_path = os.path.join(complete_path, filename)
                data['filename'].append(file_path)
                data['multiclass_label'].append(label)

    df = pd.DataFrame(data)

    # Creating binary labels
    df['binary_label'] = df['multiclass_label'].apply(lambda x: 1 if x != 0 else 0)

    return df

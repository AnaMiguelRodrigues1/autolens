import os

import numpy as np
import pandas as pd

from src.dataset.build import Dataset

def create_dataset(path_metadata: str, path_dataset: str, fold: int, binary=True) -> Dataset:
    """
    :param path_metadata: path to dataset original metadata, not the constructed for train.
    :param path_dataset:
    :param fold: metadata with folds to choose from
    :param binary: binary classification
    :return:
    """

    df = data(path_metadata, fold)
    if binary:
        df.drop("binary_cat", axis="columns", inplace=True)
        df.drop("multiclass_cat", axis="columns", inplace=True)
        df.drop("multiclass_label", axis="columns", inplace=True)
        tag = "binary"
    else:
        df.drop("binary_cat", axis="columns", inplace=True)
        df.drop("multiclass_cat", axis="columns", inplace=True)
        df.drop("binary_label", axis="columns", inplace=True)
        tag = "multiclass"

    try:
        os.mkdir("./resources")
    except FileExistsError:
        print("Resources already created")

    # Correction of path
    df['filename'] = '../../BreaKHis_v1/' + df['filename']

    # this path is now the dataset metadata for training
    df.to_csv(f"./resources/metadata_histology_{tag}.csv")

    return Dataset(f"./resources/metadata_histology_{tag}.csv", path_dataset, df.shape[0])


def load_dataset(path_metadata: str, path_dataset: str) -> Dataset:
    df = pd.read_csv(path_metadata)

    return Dataset(path_metadata, path_dataset, df.shape[0])


def data(path_metadata: str, fold: int) -> pd.DataFrame:
    df = pd.read_csv(path_metadata)

    df["binary_cat"] = df.filename.apply(lambda x: x.split("/")[3])
    df["multiclass_cat"] = df.filename.apply(lambda x: x.split('/')[-4])

    binary_names = list(np.unique(df["binary_cat"]))
    multiclass_names = list(np.unique(df["multiclass_cat"]))

    df['binary_label'] = df.binary_cat.apply(lambda x: binary_names.index(x))
    df['multiclass_label'] = df.multiclass_cat.apply(lambda x: multiclass_names.index(x))

    df = df.loc[(df['fold'] == fold)]

    df.drop("grp", axis="columns", inplace=True)
    df.drop("fold", axis="columns", inplace=True)
    df.drop("mag", axis="columns", inplace=True)

    print("Total number of images:", df.shape[0])
 
    return df

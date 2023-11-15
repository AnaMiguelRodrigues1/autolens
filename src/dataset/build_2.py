import os
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, path_dataset: str, test_size: float, valid_size: float) -> None:
        """
        :param path_dataset: path leading to dataset entrance to create metadata
        :param n_data: number of images in dataset
        :param test_size: testing dataset size in data split
        :param valid_size: validation dataset in data split
        """
        self.path_dataset = path_dataset
        self.n_data = None
        self.test_size = test_size
        self.valid_size = valid_size

    def to_metadata(self, path_dataset: str) -> tuple[pd.DataFrame, int]:
        print("Generating Metadata")
        filenames = []
        labels = []

        for root, dir, files in os.walk(path_dataset): ######
            for file in files:
                label = os.path.basename(root)
                filepath = os.path.join(root, file)
                filenames.append(filepath)
                labels.append(label)

        metadata = pd.DataFrame({'filename': filenames, 'label': label})
        n_data = len(metadata)
        self.n_data = n_data

        return metadata, n_data

    def to_path(self, test_size: float, valid_size: float, test_seed: int=1, valid_seed: int=1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Splitting Dataset")
        metadata, n_data = self.to_metadata(self.path_dataset)

        train: pd.DataFrame
        valid: pd.DataFrame
        test: pd.DataFrame

        train_and_valid, test = train_test_split(metadata, test_size=test_size, random_state=test_seed)
        train, valid = train_test_split(train_and_valid, test_size=valid_size, random_state=valid_seed)

        #del train["Unnamed: 0"]
        #del test["Unnamed: 0"]
        #del valid["Unnamed: 0"] 

        return train, test, valid

    def to_pixel(self, test_size: float, valid_size: float, test_seed: float, valid_seed: float, target_size: tuple, batch=(0, None)) -> tuple[list[np.ndarray], list[np.ndarray]]:
        print("Converting Images to Pixel Matrices")
        data = self.to_path(test_seed=test_seed, val_seed=val_seed, test_size=test_size, valid_size=valid_size)

        data_x: list[np.ndarray] = []
        data_y: list[np.ndarray] = []

        info = ["valid", "test", "train"]
        for x_m in data:
            print("Collecting data for: ", info.pop())
            # runs 3 times for train, test and validation
            if None in batch:
                n_data = x_m.shape[0]
                init = 0
            else:
                n_data = batch[1]
                init = batch[0]

            channels = 3
            x_p = np.zeros([n_data, target_size[0], target_size[1], channels], dtype=np.uint8)
            y = np.zeros([n_data], dtype=np.uint8)
            i = init
            while i < init + n_data:
                matrix = cv2.imread(x_m["filename"].iloc[i])
                # 3 channels
                for c in range(channels):
                    x_p[i-init, :, :, c] = cv2.resize(matrix[:, :, c], target_size)

                y[i-init] = x_m[list(x_m.columns)[-1]].iloc[i]

                i += 1

            data_x.append(x_p)
            data_y.append(y)

        # data[0] for x
        # data[1] for y
        # data[i][j] for train, test and validation
        return data_x, data_y




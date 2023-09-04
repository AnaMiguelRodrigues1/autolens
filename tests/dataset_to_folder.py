import sys
sys.path.insert(0, "../")

from src.dataset.build import Dataset
import random

dataset = Dataset(
    "./resources/metadata_histology_binary.csv",
    "../../BreaKHis_v1",
    7909
    )

dataset.to_folders(seed=random.randint(1,1000))



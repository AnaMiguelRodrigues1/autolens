import sys
sys.path.insert(0, "../")

from autolens.AUTOKERAS.run import main

main(
    "resources/metadata_brain_binary.csv",
    "../../brain_mri/",
    1,
    (256,256),
    0.227,
    0.1
    )


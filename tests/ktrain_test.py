import sys
sys.path.insert(0, "../")

from src.KTRAIN.run import main

main(
    "./resources/metadata_histology_binary.csv",
    "../../BreaKHis_v1/",
    1
    )


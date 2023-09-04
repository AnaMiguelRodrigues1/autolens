import sys
sys.path.insert(0, "../")

from src.KTRAIN.run import main

main(
        "./resources/metadata_histology_binary.csv",
        "/home/anarodrigues/safe_volume/BreaKHis_v1/",
        1
        )

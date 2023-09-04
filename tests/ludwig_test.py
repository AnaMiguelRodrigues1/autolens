import sys
sys.path.insert(0, "../")

from src.LUDWIG.run import main

for i in range(10):
    print(f'------------------------ Trial {i} ------------------------')
    main(
        "./resources/metadata_histology_binary.csv",
        "/home/anarodrigues/safe_volume/BreaKHis_v1/",
        1
    )

#main(
#        "./resources/metadata_histology_binary.csv",
#        "/home/anarodrigues/safe_volume/BreaKHis_v1/",
#        1        
#    )

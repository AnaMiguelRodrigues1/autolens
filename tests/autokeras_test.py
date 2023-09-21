import sys
sys.path.insert(0, "../")

from src.AUTOKERAS.run import main

for i in range(10):
    print(f'................... Trial {i} ..................')
    main(
        "resources/metadata_braintumor_multiclass.csv",
        "../../brain_mri/",
        1
        )   


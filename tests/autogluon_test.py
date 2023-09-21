import sys
sys.path.insert(0, "../")

from src.AUTOGLUON.run import main

for i in range(10):
    print(f'................... Trial {i} ..................')
    main(
        "resources/metadata_pneumonia_binary.csv",
        "../../chest_xray/",
        1
        )

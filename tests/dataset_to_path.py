from src.utils import handle_hist

FOLD = 1

created_dataset = handle_hist.create_dataset("/home/p3drx-a/Downloads/archive/Folds.csv", "TEST", FOLD)
created_train, created_test, created_valid = created_dataset.to_path()

loaded_dataset = handle_hist.load_dataset("./resources/metadata_histology_multiclass.csv", "TEST")
loaded_train, loaded_test, loaded_valid = loaded_dataset.to_path()

print(created_train.head())
print(created_test.head())
print(created_valid.head())

print(loaded_train.head())
print(loaded_test.head())
print(loaded_valid.head())

from autolens.utils import handle_hist

loaded_dataset = handle_hist.load_dataset(
    "./resources/metadata_histology_multiclass.csv",
    "../../BreaKHis_v1/")
data_x, data_y = loaded_dataset.to_pixel(batch=(128, 32), target_size=(256, 256))

print(data_x[1].shape)
print(data_y[1].shape)

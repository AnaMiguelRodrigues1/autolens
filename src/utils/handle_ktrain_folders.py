import os
import shutil

def delete_tmp_and_checkpoint(directory):
    tmp_files = []
    tmp_folders = []
    has_checkpoint = False

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path) and item.startswith("tmp"):
            tmp_files.append(item)
            os.remove(item_path)
        elif os.path.isdir(item_path):
            if item.startswith("tmp"):
                tmp_folders.append(item)
                shutil.rmtree(item_path)
            elif item == "checkpoint":
                has_checkpoint = True

    if tmp_files:
        print("... files 'tmp' deleted")
    if tmp_folders:
        print("... folders 'tmp' deleted")
    if not (tmp_files or tmp_folders or has_checkpoint):
        print('... no files/folders to delete')


# Ludwig generates directories with long and scattered names, which can make it challenging to manage and organize the output files effectively

import os
import shutil

def handle_directories_from_folder():
    print('... deleting ludwig previous subfolders')
    items_to_delete = ['/model','/resources','/description.json','/training_statistics.json']
    path='resources/ludwig'
    for item in items_to_delete:
        if os.path.exists(path+item):
            if os.path.isdir(path+item):
                shutil.rmtree(path+item)
                print(f"... folder '{item}' deleted.")
            else:
                os.remove(path+item)
                print(f"... file '{item}' deleted.")
        else:
            print(f"... '{item}' does not exist.")

def add_directories_to_folder():
    print('... organizing ludwig folder')
    if os.path.exists('resources/ludwig'):
        if os.path.exists('resources/ludwig/ludwig_metadata'):
            shutil.rmtree('resources/ludwig/ludwig_metadata')
            os.mkdir('resources/ludwig/ludwig_metadata')
        else:
            os.mkdir('resources/ludwig/ludwig_metadata')
        folders = os.listdir('./')

        for folder in folders:
            if folder.split('.')[-1]=='json' or folder.split('.')[-1]=='hdf5':
                shutil.copy(folder, 'resources/ludwig/ludwig_metadata')
                os.remove(folder)
        print('... done')
    else:
        print('... ludwig folder not found')


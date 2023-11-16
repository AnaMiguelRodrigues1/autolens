# If I leave the Autokeras folder from previous experiences, it will retrive that information
# To isolate cases -> Delete folder each time

import os
import shutil

def replace_classifier_folder():
    if os.path.exists('resources/autokeras'):
        print('... deleting autokeras previous folder')
        shutil.rmtree('resources/autokeras')
    else:
        print('... saving autokeras folder')

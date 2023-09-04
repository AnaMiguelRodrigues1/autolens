# Autogluon does not overwrite the previous classifier, like Autokeras does

import os
import shutil

def replace_classifier_folder():
    if os.path.exists('resources/autogluon'):
        print('... deleting autogluon previous folder')
        shutil.rmtree('resources/autogluon')
    else:
        print('... saving autogluon folder')



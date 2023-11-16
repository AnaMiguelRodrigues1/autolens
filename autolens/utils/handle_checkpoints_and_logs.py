import os 
import shutil

class Checkpoints_and_Logs():
    def __init__(self, system: str, clean: bool):
        """
        :param system: AutoML system choosed for implementation
        :param clean: clean checkpoints and logs generated
        """
        self.system = system
        self.clean = clean

    def manage_files(self, system:str, clean: bool):
        print('Cleaning checkpoints and logs')
        if clean == 0:
            continue
        else:
            items_to_delete = ['/model','/resources','/description.json','/training_statistics.json']
            path='./resources/ludwig'
            for item in items_to_delete:
                if os.path.exists(path+item):
                    if os.path.isdir(path+item):
                        shutil.rmtree(path+item)
                else:
                    os.remove(path+item)
                else:
                    print(f"... '{item}' does not exist.")
    
    def handle_metadata(self, system: str, clean: bool):
        # Improve here!!!
        if clean == 0:
            if os.path.exists('resources/autogluon'):
                print('... deleting autogluon previous folder')
                shutil.rmtree('resources/autogluon')
            else:
                print('... saving autogluon folder')
        else:
            continue

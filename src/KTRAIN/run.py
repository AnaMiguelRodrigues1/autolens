import time

import ktrain
from ktrain import vision as vis

from src.utils.handle_hist import load_dataset
from src.utils.handle_results import save_results

def main(
        path_metadata: str,
        path_dataset: str,
        steps: int,
        target_size=(255, 255)):
        
        """
        :param path_metadata:
        :param path_dataset:
        :param steps:
        :param target_size:
        :return:
        """
        
        print('Time --> Start')
        start_time = time.time()

        dataset = load_dataset(path_metadata, path_dataset)
        n_data = dataset.n_data

        print('Building Archiecture')
        model = vis.image_classifier(
                'pretrained_resnet50', 
                trn, 
                val, 
                freeze_layers=15)
        
        histories = []
        scores = []
        predictions = []
        prob_predictions = []

        new_path = dataset.to_folders()
        print('Loading the Data')
        (trn, val, preproc) = vis.images_from_folder(
                datadir=new_path,
                data_aug = vis.get_data_aug(horizontal_flip=True),
                train_test_names=['train', 'valid'])

        print('Fitting Model')
        learner = ktrain.get_learner(
                model=model, 
                train_data=trn, 
                val_data=val, 
                use_multiprocessing=False,
                batch_size=64)

        learnerlr_find(max_epochs=5)

        learner.fit_onecycle(1e-4,1)
        print('I have completed')


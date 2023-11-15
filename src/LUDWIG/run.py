import time
import random

from ludwig.api import LudwigModel
import logging

from src.dataset.build_2 import Dataset
from src.utils.create_resources_folder import resources
from src.utils.handle_results import save_results
from src.utils.handle_ludwig_folder import handle_directories_from_folder, add_directories_to_folder
from src.utils import handle_ludwig_metrics

def main(path_dataset: str,
    steps=1,
    target_size=(255,255),
    test_size=0.2,
    valid_size=0.1):

    """
    :param path_dataset:
    :param steps:
    :param target_size:
    :param test_size:
    :param valid_size:
    :return:
    """
    
    resources()

    print('Time --> Start')
    start_time = time.time()

    # Handling folders created by Ludwig
    handle_directories_from_folder()
    add_directories_to_folder()

    print('Loading Data')
    dataset = Dataset(path_dataset=path_dataset, 
                    test_size=test_size, 
                    valid_size=valid_size)
    train, test, valid = dataset.to_path(test_size=test_size,
                                        valid_size=valid_size,
                                        test_seed=random.randint(1, 10000),
                                        valid_seed=random.randint(1, 10000))

    print('Building Architecture')
    config = {
        'input_features': [
            {
            'name': 'filename',
            'type': 'image',
            'preprocessing': {
            'num_processes': 4
                },
                'encoder': 'stacked_cnn'
            }
        ],
        'output_features': [
            {
            'name': 'label', # multiclass_label
            'type': 'binary'  # category 
            }
        ],
        'training': {
            'epochs':5 
            },
        'hyperopt': {
            'parameters': {},
            'executor': {'num_samples': 16},
            'search_alg': {
            'type': 'variant_generator',
            'random_state': random.randint(1, 10000),
            'n_startup_jobs': 10},
            'goal': 'maximize',
            'metric': 'roc_auc',
            'output_feature': 'binary_label', # multiclass_label
            }
        }

    model = LudwigModel(config, logging_level=logging.INFO)
    
    print('Fitting Model')
    train_stats, preprocessed_data, output_directory = model.train(
            training_set=train,validation_set=valid,
            experiment_name='resources/ludwig',
            model_name='Model',
            model_resume_path='resources/ludwig',
            output_directory='resources/ludwig',
            random_seed=random.randint(1, 10000),
            skip_save_training_description=False,
            skip_save_training_statistics=False
            )

    histories={'train':train_stats['training'],
            'valid':train_stats['validation']}

    print('Evaluating Model')
    test_stats, predictions, output_directory = model.evaluate(test)
    scores = test_stats['label']

    print('Predictions')
    predictions_lw, output_directory = model.predict(test)

    # Predicted labels
    predictions_raw = predictions_lw["label_predictions"].tolist()
    y_pred = [int(i) for i in predictions_raw]

    # Probabilities of the predicted labels
    y_prob = predictions_lw["label_probability"].tolist()

    # Actual labels
    y_test = test['label'].tolist()

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('Time(min):', round(elapsed_time/60,2))

    print('Saving Results...')
    all_results = save_results(histories, scores, y_pred, y_prob, y_test, elapsed_time)
    all_results.to_csv('resources/ludwig_results.csv', mode='a', index=False)
    print('Results Ready!')


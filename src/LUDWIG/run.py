import time
import random

from ludwig.api import LudwigModel
import logging

from src.utils.handle_hist import load_dataset
from src.utils.create_resources_folder import resources
from src.utils.handle_results import save_results
from src.utils.handle_ludwig_folder import handle_directories_from_folder, add_directories_to_folder
from src.utils.handle_ludwig_metrics import F1Score

def main(
        path_metadata: str,
        path_dataset: str,
        steps: int
        ):
    """
    :param path_metadata:
    :param path_dataset:
    :param steps:
    :param target_size:
    :return:
    """

    resources()

    print('Time --> Start')
    start_time = time.time()

    # Handling folders created by Ludwig
    handle_directories_from_folder()
    add_directories_to_folder()

    print('Registering F1-Score')
    F1Score(num_classes=2) 

    print('Loading Data')
    dataset = load_dataset(path_metadata, path_dataset)
    n_data = dataset.n_data
    train, test, valid = dataset.to_path()
    
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
            'name': 'binary_label',
            'type': 'binary'
            }
        ],
        'training': {
            'epochs':25  
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
            'output_feature': 'binary_label',
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

    print('Evaluating Model')
    test_stats, predictions, output_directory = model.evaluate(test)

    print('Predictions')
    predictions_lw, output_directory = model.predict(test)

    # Predicted labels
    predictions_boolean = predictions_lw["binary_label_predictions"].tolist()
    y_pred = [1 if p else 0 for p in predictions_boolean]

    # Probabilities of the predicted labels
    y_prob = predictions_lw["binary_label_probability"].tolist()

    # Actual labels
    y_test = test['binary_label'].tolist()

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('Time:', elapsed_time)

    print('Saving Results...')
    all_results = save_results(raw_data, test_stats, y_pred, y_prob, y_test, elapsed_time)
    all_results.to_csv('resources/ludwig_results.csv', mode='a', index=False)
    print('Results Ready!')


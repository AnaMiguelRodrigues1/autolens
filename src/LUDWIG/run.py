import time
import random

from ludwig.api import LudwigModel
import logging

from src.utils.handle_hist import load_dataset
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

    # Enconders
    # enc_list=['stacked_cnn', '_resnet_legacy', 'mlp_mixer', '_vit_legacy', 'alexnet', 'convnext', 'densenet', 'efficientnet', 'googlenet', 'inceptionv3', 'maxvit', 'mnasnet', 'mobilenetv2', 'mobilenetv3', 'regnet', 'resnet', 'resnext', 'shufflenet_v2', 'squeezenet', 'swin_transformer', 'vit', 'vgg', 'wide_resnet']
    encoder = 'densenet'

    print('Time --> Start')
    start_time = time.time()

    dataset = load_dataset(path_metadata, path_dataset)
    n_data = dataset.n_data

    # Handling the folders create
    handle_directories_from_folder()
    add_directories_to_folder()
        
    print('Building Architecture')
    config = {
        'input_features': [
            {
            'name': 'filename',
            'type': 'image',
            'preprocessing': {
            'num_processes': 4
                },
                'encoder': encoder
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
    print('Model', model)

    histories = []
    scores = []
    predictions = []
    prob_predictions = []

    #train, test, valid = dataset.to_path()
    train, test, valid = dataset.to_path(test_seed=random.randint(1, 10000), val_seed=random.randint(1, 10000))

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

    add_directories_to_folder()

    # Deconstructing Ludwig Format
    raw_data = {}
    count = 0
    for i in train_stats:
        if len(i)==0:
            continue
        else:
            count+=1
            if count==1:
                raw_data['train'] = i 
            else:
                raw_data['valid'] = i

    print(len(raw_data))

    print('Evaluating Model')
    test_stats, predictions, output_directory = model.evaluate(test)
    print('test_stats', len(test_stats))

    print('Predictions')
    predictions_lw, output_directory = model.predict(test)

    # Predicted Labels
    predictions_boolean = predictions_lw["binary_label_predictions"].tolist()
    predictions = [1 if p else 0 for p in predictions_boolean]

    # Probabilities of Predicted Labels
    prob_predictions = predictions_lw["binary_label_probability"].tolist()

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('TIME', elapsed_time)

    print('Saving Results')
    y_test = test['binary_label'].tolist()

    all_results = save_results(raw_data, test_stats, predictions, prob_predictions, y_test, elapsed_time)

    all_results.to_csv('resources/ludwig_results.csv', mode='a', index=False)
    print('Results Ready!')


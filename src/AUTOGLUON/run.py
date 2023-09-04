import time
import random

from autogluon.multimodal import MultiModalPredictor

from src.utils.handle_hist import load_dataset
from src.utils.handle_autogluon_folder import replace_classifier_folder
from src.utils.handle_results import save_results

def main(path_metadata: str,
        path_dataset: str,
        steps: int):
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
    train, test, valid = dataset.to_path(test_seed=random.randint(1, 10000), val_seed=random.randint(1, 10000)) 
    #train, test, valid = dataset.to_path()

    replace_classifier_folder()
    print('Building Architecture')
    classifier = MultiModalPredictor(
            label="binary_label",
            problem_type='binary',
            path='resources/autogluon',
            verbosity=4,
            eval_metric='roc_auc', 
            validation_metric='roc_auc')

    print('Fitting Model')
    hyperparameter_tune_kwargs = {
        "num_trials": 2}

    hyperparameters = {
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0, # otherwise -> ssh error (insuficient memnory in docker container)
        "env.num_gpus": 1}

    history = classifier.fit(
            train_data=train,
            tuning_data=valid,
            column_types={
                "filename": "image_path",
                "binary_label": "binary"
                },
            seed=random.randint(1, 10000),
            presets='medium_quality',
            hyperparameters=hyperparameters,
            #hyperparameter_tune_kwargs=hyperparameter_tune_kwargs
            time_limit=1560 # in seconds
            )

    # The current version does not record the loss or accuracy for each epoch
    histories = classifier.fit_summary(verbosity=4, show_plot=True)
    # Model is saved automatically -> loaded_predictor = MultiModalPredictor.load(model_path)

    print('Evaluating Model')
    scores = classifier.evaluate(data=test, metrics=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "mcc"])

    print('Predictions')
    predictions = classifier.predict(test).tolist() 
    
    # Probabilities of labels
    prob_predictions = classifier.predict_proba(test)
    prob_predictions = prob_predictions[[0, 1]].max(axis=1)
    prob_predictions = prob_predictions.tolist()

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('TIME', elapsed_time)

    print('Saving Results')
    y_test = test['binary_label'].tolist()
    all_results = save_results(histories, scores, predictions, prob_predictions, y_test, elapsed_time)
    
    all_results.to_csv('resources/autogluon_results.csv', mode='a', index=False)
    print('Results Ready!')

  


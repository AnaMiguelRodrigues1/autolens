import time
import random

from autogluon.multimodal import MultiModalPredictor

from src.utils.handle_hist import load_dataset
from src.utils.create_resources_folder import resources
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
    resources()

    print('Time --> Start')
    start_time = time.time()

    # Handling folders created by AutoGluon
    replace_classifier_folder()

    print('Loading Data')
    dataset = load_dataset(path_metadata, path_dataset)
    n_data = dataset.n_data
    train, test, valid = dataset.to_path()

    print('Building Architecture')
    classifier = MultiModalPredictor(
            label="binary_label",
            problem_type='binary',
            path='resources/autogluon',
            verbosity=4,
            eval_metric='roc_auc',
            validation_metric='roc_auc'
            )

    print('Fitting Model')
    history = classifier.fit(
            train_data=train,
            tuning_data=valid,
            column_types={
                "filename": "image_path",
                "binary_label": "binary"
                },
            seed=random.randint(1, 10000),
            presets='medium_quality',
            hyperparameters={
                "env.num_workers": 0,
                "env.num_workers_evaluation": 0, # otherwise raises ssh error 
                "env.num_gpus": 1
                },
            time_limit=600 # in seconds
            )

    histories = classifier.fit_summary(verbosity=4, show_plot=True)

    # Model is automatically saved
    # loaded_predictor = MultiModalPredictor.load(model_path)

    print('Evaluating Model')
    scores = classifier.evaluate(
            data=test, 
            metrics=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "mcc"])

    print('Predictions')
    # Predicted labels
    y_pred = classifier.predict(test).tolist() 
    
    # Probabilities of the predicted labels
    prob = classifier.predict_proba(test)
    y_prob = (prob[[0, 1]].max(axis=1)).tolist()

    # Actual labels
    y_test = test['binary_label'].tolist()

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('Time:', elapse_time)

    print('Saving Results...')
    all_results = save_results(histories, scores, y_pred, y_prob, y_test, elapsed_time)
    all_results.to_csv('resources/autogluon_results.csv', mode='a', index=False)
    print('Results Ready!')

  


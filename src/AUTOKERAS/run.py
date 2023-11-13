import tensorflow as tf
import numpy as np
import time
import keras
from keras_tuner import Objective
import random

from autokeras import ImageClassifier

from src.utils import handle_dataset
from src.utils.create_resources_folder import resources
from src.utils.handle_autokeras_folder import replace_classifier_folder
from src.utils.handle_results import save_results

def main(
        path_metadata: str,
        path_dataset: str,
        steps: int,
        target_size: tuple,
        test_size: float, 
        valid_size: float 
        ):

    """
    :param path_metadata:
    :param path_dataset:
    :param steps:
    :param target_size:
    :param test_size:
    :param valid_size
    """

    # To make sure resources folder is created
    resources()

    print('Time --> Start')
    start_time = time.time()

    # Handling folder created by AutoKeras
    replace_classifier_folder()

    print('Loading Data')
    dataset = handle_dataset.check(path_dataset)
    n_data = dataset.n_data
    data = dataset.to_pixel(test_seed=random.randint(1, 10000), 
                            val_seed=random.randint(1, 10000),
                            test_size=test_size,
                            valid_size=valid_size)

    print('Building Architecture')
    model = ImageClassifier(
        project_name='autokeras_model',
        directory='resources/autokeras',
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
        # F1-Score not available in this keras version
        max_trials=1,
        objective=Objective('val_auc', direction="max") 
        )

    histories = []
    scores = []
    predictions = []
    prob_predictions = []
    if steps == 1:
        print("Training step: ", steps)
        print('Fitting Model')
        history = model.fit(
            data[0][0],
            data[1][0],
            validation_data=(data[0][2], data[1][2]), # or validation_split=0.15,
            epochs=1
            ) 

        histories.append(history)

        print('Evaluating Model')
        score = model.evaluate(data[0][1], data[1][1])
        scores.append(score)

    else: # in case of bad memory management
        batch_size = round(n_data/steps)
        batch = 0
        for i in range(steps):
            if i == steps-1 and batch_size is not None:
                # last step
                batch_size = n_data - batch

            print("Training step: ", i+1)
            data = dataset.to_pixel(batch=(batch, batch_size), target_size=target_size)
            history = model.fit(
                data[0][0],
                data[1][0],
                validation_data=(data[0][2], data[1][2]), # or validation_split=0.15,
                epochs=1
                )
            
            histories.append(history)

            print('Evaluating Model')
            score = model.evaluate(data[0][1], data[1][1])
            scores.append(score)

            batch += batch_size

    # Model is not saved automatically
    best_model = model.export_model()
    best_model.save("resources/autokeras/autokeras_model.keras")
            
    print('Predictions')
    # Predicted labels
    y_predic = (model.predict(data[0][1])).astype("int32")
    y_predic = y_predic.flatten()

    predictions.append(y_predic)
    predictions = predictions[0].tolist()

    # Probabilities of the predicted labels
    for i in range(0, len(data[0][1]), 32):
        y_prob_int = model.export_model()(data[0][1][i:i+32]) # works as tensorflow keras model
        y_prob_int = np.array(y_prob_int).flatten().tolist()
        y_prob_int_rounded = [round(prob, 4) for prob in y_prob_int]
        prob_predictions.extend(y_prob_int_rounded)

    # Actual labels
    y_test = data[1][1].tolist()
    print('true labels', len(y_test))

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    print('Time:', elapsed_time)

    print('Saving Results...')
    all_results = save_results(histories, scores, predictions, prob_predictions, y_test, elapsed_time)
    all_results.to_csv('resources/autokeras_results.csv', mode='a', index=False)
    print('Results Ready!')


from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score

# Define your custom callback class
class CustomMetricsCallback(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        val_features = self.val_data[0]  
        val_labels = self.val_data[1]

        val_predictions = self.model.predict(val_features)

        # Calculate the F1 score
        f1 = f1_score(
            np.argmax(val_labels, axis=1),
            np.argmax(val_predictions, axis=1),
            average='weighted'
        )

        # Add the F1 score to the logs
        logs["custom_f1_score"] = f1


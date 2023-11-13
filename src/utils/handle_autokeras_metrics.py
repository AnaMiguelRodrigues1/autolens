import tensorflow as tf

num_classes = 4

class MacroPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='macro_precision', **kwargs):
        super(MacroPrecision, self).__init__(name=name, **kwargs)
        self.precision_metrics = [tf.keras.metrics.Precision() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i, metric in enumerate(self.precision_metrics):
            metric.update_state(y_true[:, i], y_pred[:, i])

    def result(self):
        total_precision = 0
        for metric in self.precision_metrics:
            total_precision += metric.result()
        return total_precision / len(self.precision_metrics)

class MacroRecall(tf.keras.metrics.Metric):
    def __init__(self, name='macro_recall', **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.recall_metrics = [tf.keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i, metric in enumerate(self.recall_metrics):
            metric.update_state(y_true[:, i], y_pred[:, i])

    def result(self):
        total_recall = 0
        for metric in self.recall_metrics:
            total_recall += metric.result()
        return total_recall / len(self.recall_metrics)

class MacroAUC(tf.keras.metrics.Metric):
    def __init__(self, name='macro_auc', **kwargs):
        super(MacroAUC, self).__init__(name=name, **kwargs)
        self.auc_metrics = [tf.keras.metrics.AUC() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i, metric in enumerate(self.auc_metrics):
            metric.update_state(y_true[:, i], y_pred[:, i])

    def result(self):
        total_auc = 0
        for metric in self.auc_metrics:
            total_auc += metric.result()
        return total_auc / len(self.auc_metrics)

from ludwig.constants import ACCURACY, BINARY, CATEGORY, CATEGORY_DISTRIBUTION, MAXIMIZE, PREDICTIONS
from ludwig.modules.metric_registry import register_metric
from ludwig.modules.metric_modules import LudwigMetric, MulticlassAccuracy
import torch
from torchmetrics.classification import MulticlassF1Score

# This Class can be used to register other metrics
print('Registering the F1_Score Metric')
@register_metric("my_f1", [CATEGORY, CATEGORY_DISTRIBUTION], MAXIMIZE, PREDICTIONS)
class Myf1score(LudwigMetric):
        def __init__(self, num_classes: int, **kwargs):
            super().__init__()
            self.f1 = MulticlassF1Score(num_classes=num_classes)

        def update(self, preds, target) -> None:
            self.f1.update(preds, target)

        def compute(self, **kwargs):
            return self.f1.compute()
            

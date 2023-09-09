from ludwig.constants import ACCURACY, BINARY, CATEGORY, CATEGORY_DISTRIBUTION, MAXIMIZE, PREDICTIONS
from ludwig.modules.metric_registry import register_metric
from ludwig.modules.metric_modules import LudwigMetric, MulticlassAccuracy
import torch
from torchmetrics.classification import MulticlassF1Score

# This Class can be used to register other metrics
print('Registering the F1_Score Metric')
@register_metric("F1 Score", [CATEGORY, CATEGORY_DISTRIBUTION], MAXIMIZE, PREDICTIONS)
class F1Score(MulticlassF1Score, LudwigMetric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes)

    def update(self, preds, target) -> None:
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=1)
            super().update(preds, target.type(torch.long))

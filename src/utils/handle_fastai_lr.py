# https://forums.fast.ai/t/automated-learning-rate-suggester/44199

from fastai.learner import Learner
import matplotlib.pyplot as plt
import numpy as np

def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    
    #Run the Learning Rate Finder
    model.lr_find()
    
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs
    
    # Find the index where the loss is lowest before the loss spike
    min_loss_idx = np.argmin(losses)
    r_idx = min_loss_idx
    l_idx = max(min_loss_idx - lr_diff, 0)
    
    while (l_idx >= 0) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
        r_idx -= 1
        l_idx -= 1

    # Calculate the learning rate to use
    lr_to_use = lrs[l_idx] * adjust_value 

    return lr_to_use

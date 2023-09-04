import numpy as np

def find_lr_range(losses, lrs):

    # Calculate gradients of the loss curve
    loss_grad = np.gradient(losses)
    
    # lr_min
    min_grad_idx = np.argmin(loss_grad)
    print(min_grad_idx)
    lr_min = lrs[min_grad_idx]
    print(lr_min)

    # Find index of the minimum loss
    min_loss_idx = np.argmin(losses)

    lr_min_loss = lrs[min_loss_idx]

    lr_max = lr_min_loss / 10

    return lr_min, lr_max

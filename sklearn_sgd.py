"""Optimization methods for MLP
"""

# Authors: Jiyuan Qian <jq401@nyu.edu>
# License: BSD 3 clause

# Adaptation of sklearn.neural_network._optimizers
# Heavily streamlined by Heather Macbeth for pedagogical purposes

import numpy as np

class OptimizerStatus:
    def __init__(self, params, learning_rate_init=0.1, **kwargs):
        self.learning_rate = float(learning_rate_init)
        self.velocities = [np.zeros_like(param) for param in params]

def get_updates(optimizer, grads, lr_schedule='constant',
              momentum=0., nesterov=False, power_t=0.5, **kwargs):
    """Get the values used to update params with given gradients
    """
    # note that this is an update for gradient *ascent*, not gradient *descent*!!
    updates = [momentum * velocity + optimizer.learning_rate * grad
                for velocity, grad in zip(optimizer.velocities, grads)]
    optimizer.velocities = updates

    if nesterov:
        updates = [momentum * velocity + optimizer.learning_rate * grad
                    for velocity, grad in zip(optimizer.velocities, grads)]

    if lr_schedule == 'invscaling':
        optimizer.learning_rate /= power_t

    return updates

def trigger_stopping(optimizer, lr_schedule='constant', **kwargs):
    if lr_schedule != 'adaptive':
        return True

    if optimizer.learning_rate <= 1e-6:
        return True

    optimizer.learning_rate /= 5.
    return False
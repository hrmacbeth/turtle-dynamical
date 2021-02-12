# Adapted from scikit-learn.examples.classification.plot_classifier_comparison
#
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numbers
import array
from collections.abc import Iterable

import numpy as np
from scipy import linalg
import scipy.sparse as sp

from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.datasets import make_classification

def make_linearly_separable(n_samples=100, scale=1, class_sep=1, noise=0, random_state_1=None, random_state_2=None):
	"""Make a synthetic dataset approximately classified by a hyperplane."""

    X, y = make_classification(n_samples=n_samples, scale=scale, shift=[-class_sep, -class_sep],
          class_sep=class_sep,n_features=2, n_redundant=0, n_informative=2,
                           random_state=random_state_1, n_clusters_per_class=1)
    rng = np.random.RandomState(random_state_2)

    print(noise)
    X += rng.uniform(2 * noise, size=X.shape)
    X -= noise * np.ones(X.shape)

    return X, y

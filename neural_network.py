"""Multi-layer Perceptron
"""

# Authors: Issam H. Laradji <issam.laradji@gmail.com>
#          Andreas Mueller
#          Jiyuan Qian
# License: BSD 3 clause

# Adaptation of sklearn.neural_network._multilayer_perceptron
# Heavily streamlined by Heather Macbeth for pedagogical purposes

import numpy as np

from abc import ABCMeta, abstractmethod
import warnings

import scipy.optimize
import config as cfg

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils import column_or_1d
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklearn.utils.multiclass import _check_partial_fit_first_call, unique_labels
from sklearn.utils.multiclass import type_of_target


exec(f'from {cfg.optimizer_file} import OptimizerStatus, get_updates, trigger_stopping')


class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):
    """Base class for MLP classification and regression.
    Warning: This class should not be used directly.
    Use derived classes instead.
    .. versionadded:: 0.18
    """

    @abstractmethod
    def __init__(self, hidden_layer_sizes, activation,
                 alpha,
                 loss, random_state, tol, verbose,
                 warm_start,
                 n_iter_no_change):
        self.activation = activation
        self.alpha = alpha
        self.loss = loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_iter_no_change = n_iter_no_change


    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.
        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],
                                                 self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])

        return activations

    def _forward_pass_fast(self, X):
        """Predict using the trained model
        This is the same as _forward_pass but does not record the activations
        of all layers and only returns the last layer's activation.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        """
        X = self._validate_data(X, accept_sparse=['csr', 'csc'], reset=False)

        # Initialize first layer
        activation = X

        # Forward propagate
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)

        return activation

    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.
        This function does backpropagation for the specified one layer.
        """
        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)


    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        inplace_derivative = DERIVATIVES[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads

    def _initialize(self, y, layer_units, dtype):
        # set all attributes, allocate weights etc for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for multi class
        if self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation_ = 'softmax'
        # Output for binary class and multi-label
        else:
            self.out_activation_ = 'logistic'

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i],
                                                        layer_units[i + 1],
                                                        dtype)
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self.loss_curve_ = []
        self._no_improvement_count = 0
        self.best_loss_ = np.inf

    def _init_coef(self, fan_in, fan_out, dtype):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init

    def _fit(self, X, y, **kwargs):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_sizes)
        first_pass = (not hasattr(self, 'coefs_') or
                      (not self.warm_start))

        X, y = self._validate_input(X, y, reset=first_pass)

        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(y, layer_units, X.dtype)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
                      for n_fan_in_,
                      n_fan_out_ in zip(layer_units[:-1],
                                        layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_, dtype=X.dtype)
                           for n_fan_out_ in
                           layer_units[1:]]

        # Run the Stochastic optimization solver
        self._fit_stochastic(X, y, activations, deltas, coef_grads,
                             intercept_grads, layer_units, **kwargs)

        return self

    def _validate_hyperparameters(self):
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0, got %s." % self.alpha)
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be > 0, got %s."
                             % self.n_iter_no_change)

        # raise ValueError if not registered
        if self.activation not in ACTIVATIONS:
            raise ValueError("The activation '%s' is not supported. Supported "
                             "activations are %s."
                             % (self.activation, list(sorted(ACTIVATIONS))))

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads, layer_units, max_iter=200, **kwargs):

        params = self.coefs_ + self.intercepts_

        optimizer = OptimizerStatus(params, **kwargs)

        X_val = None
        y_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        try:
            for it in range(max_iter):
                activations[0] = X
                loss, coef_grads, intercept_grads = self._backprop(
                    X, y, activations, deltas,
                    coef_grads, intercept_grads)

                # update weights
                grads = coef_grads + intercept_grads
                updates = get_updates(optimizer, grads, **kwargs)
                for param, update in zip(params, updates):
                    param -= update

                self.n_iter_ += 1
                self.loss_ = loss

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_,
                                                         self.loss_))

                # update no_improvement_count based on training loss
                self._update_no_improvement_count(X_val, y_val)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    msg = ("Training loss did not improve more than tol=%f"
                           " for %d consecutive epochs." % (
                               self.tol, self.n_iter_no_change))

                    is_stopping = trigger_stopping(optimizer, **kwargs)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if self.n_iter_ == max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % max_iter, ConvergenceWarning)
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

    def _update_no_improvement_count(self, X_val, y_val):
        if self.loss_curve_[-1] > self.best_loss_ - self.tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        if self.loss_curve_[-1] < self.best_loss_:
            self.best_loss_ = self.loss_curve_[-1]

    def fit(self, X, y, **kwargs):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : returns a trained MLP model.
        """
        return self._fit(X, y, **kwargs)

    @property
    def num_parameters(self):
        hls = self.hidden_layer_sizes
        if len(hls) == 0:
            return 3
        num_params = 3 * hls[0]
        for i in range(len(hls) - 1):
            num_params += (hls[i] + 1) * hls[i + 1]
        num_params += hls[len(hls) - 1] + 1
        return num_params

class MLPClassifier(ClassifierMixin, BaseMultilayerPerceptron):
    """Multi-layer Perceptron classifier.
    This model optimizes the log-loss function using stochastic
    gradient descent.
    .. versionadded:: 0.18
    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.
        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant' is a constant learning rate given by
          'learning_rate_init'.
        - 'invscaling' gradually decreases the learning rate at each
          time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)
        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, the current learning rate is divided by 5.
    learning_rate_init : double, default=0.001
        The initial learning rate used. It controls the step-size
        in updating the weights.
    power_t : double, default=0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'.
    max_iter : int, default=200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. Note that this determines 
        the number of epochs (how many times each data point will be used), not the 
        number of gradient steps.
    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization, and batch sampling.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    tol : float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.
    verbose : bool, default=False
        Whether to print progress messages to stdout.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.
    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1.
    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum. Only used when
        momentum > 0.
    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet ``tol`` improvement.
        .. versionadded:: 0.20
    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.
    loss_ : float
        The current loss computed with the loss function.
    best_loss_ : float
        The minimum loss reached by the solver throughout fitting.
    loss_curve_ : list of shape (`n_iter_`,)
        The ith element in the list represents the loss at the ith iteration.
    t_ : int
        The number of training samples seen by the solver during fitting.
    coefs_ : list of shape (n_layers - 1,)
        The ith element in the list represents the weight matrix corresponding
        to layer i.
    intercepts_ : list of shape (n_layers - 1,)
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.
    n_iter_ : int
        The number of iterations the solver has ran.
    n_layers_ : int
        Number of layers.
    n_outputs_ : int
        Number of outputs.
    out_activation_ : str
        Name of the output activation function.
    Examples
    --------
    >>> from sklearn.neural_network import MLPClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    ...                                                     random_state=1)
    >>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    >>> clf.predict_proba(X_test[:1])
    array([[0.038..., 0.961...]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    Notes
    -----
    MLPClassifier trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.
    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.
    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values.
    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.
    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.
    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).
    """
    @_deprecate_positional_args
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *,
                 alpha=0.0001,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, n_iter_no_change=10):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, alpha=alpha,
            loss='log_loss',
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start,
            n_iter_no_change=n_iter_no_change)

    def _validate_input(self, X, y, reset):
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'],
                                   multi_output=True,
                                   dtype=(np.float64, np.float32),
                                   reset=reset)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        # Matrix of actions to be taken under the possible combinations:
        # The cases are already grouped into the respective if blocks below.
        #
        # warm_start classes_ def  action
        #        0         0        define classes_
        #        1         0        define classes_
        #        0         1        redefine classes_
        #        1         1        check compat warm_start
        #
        # Note the reliance on short-circuiting here, so that the second
        # or part implies that classes_ is defined.
        if (
            (not hasattr(self, "classes_")) or
            (not self.warm_start)
        ):
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
        else:
            classes = unique_labels(y)
            if self.warm_start:
                if set(classes) != set(self.classes_):
                    raise ValueError(
                        f"warm_start can only be used where `y` has the same "
                        f"classes as in the previous call to fit. Previously "
                        f"got {self.classes_}, `y` has {classes}"
                    )
            elif len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                raise ValueError(
                    f"`y` has classes not in `self.classes_`. "
                    f"`self.classes_` has {self.classes_}. 'y' has {classes}."
                )

        # This downcast to bool is to prevent upcasting when working with
        # float32 data
        y = self._label_binarizer.transform(y).astype(bool)
        return X, y

    def predict(self, X):
        """Predict using the multi-layer perceptron classifier
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        check_is_fitted(self)
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return self._label_binarizer.inverse_transform(y_pred)


    def predict_proba(self, X):
        """Probability estimates.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

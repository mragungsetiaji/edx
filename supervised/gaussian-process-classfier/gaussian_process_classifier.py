import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import erf, expit

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.gaussian_process.kernels \
    import RBF, CompoundKernel, ConstantKernel as C
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                  128.12323805, -2010.49422654])[:, np.newaxis]

class GaussianProcessClassifier(BaseEstimator, ClassifierMixin):
    """Gaussian process classification (GPC) based on Laplace approximation.
    The implementation is based on Algorithm 3.1, 3.2, and 5.1 of
    Gaussian Processes for Machine Learning (GPML) by Rasmussen and
    Williams.
    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.
    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.
    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.
    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::
            'fmin_l_bfgs_b'
    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.
    max_iter_predict : int, optional (default: 100)
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.
    warm_start : bool, optional (default: False)
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization.
    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    multi_class : string, default : "one_vs_rest"
        Specifies how multi-class classification problems are handled.
        Supported are "one_vs_rest" and "one_vs_one". In "one_vs_rest",
        one binary Gaussian process classifier is fitted for each class, which
        is trained to separate this class from the rest. In "one_vs_one", one
        binary Gaussian process classifier is fitted for each pair of classes,
        which is trained to separate these two classes. The predictions of
        these binary predictors are combined into multi-class predictions.
        Note that "one_vs_one" does not support predicting probability
        estimates.
    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    Attributes
    ----------
    kernel_ : kernel object
        The kernel used for prediction. In case of binary classification,
        the structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters. In case of multi-class
        classification, a CompoundKernel is returned which consists of the
        different kernels used in the one-versus-rest classifiers.
    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``
    classes_ : array-like, shape = (n_classes,)
        Unique class labels.
    n_classes_ : int
        The number of classes in the training data
    .. versionadded:: 0.18
    """
    def __init__(self, kernel=None, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, max_iter_predict=100,
                 warm_start=False, copy_X_train=True, random_state=None,
                 multi_class="one_vs_rest", n_jobs=1):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit Gaussian process classification model
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples,)
            Target values, must be binary
        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y, multi_output=False)

        self.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(
            self.kernel, self.optimizer, self.n_restarts_optimizer,
            self.max_iter_predict, self.warm_start, self.copy_X_train,
            self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError("GaussianProcessClassifier requires 2 or more "
                             "distinct classes. Only class %s present."
                             % self.classes_[0])
        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = \
                    OneVsRestClassifier(self.base_estimator_,
                                        n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = \
                    OneVsOneClassifier(self.base_estimator_,
                                       n_jobs=self.n_jobs)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class)

        self.base_estimator_.fit(X, y)

        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = np.mean(
                [estimator.log_marginal_likelihood()
                 for estimator in self.base_estimator_.estimators_])
        else:
            self.log_marginal_likelihood_value_ = \
                self.base_estimator_.log_marginal_likelihood()

        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self, ["classes_", "n_classes_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
        Returns
        -------
        C : array-like, shape = (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        check_is_fitted(self, ["classes_", "n_classes_"])
        if self.n_classes_ > 2 and self.multi_class == "one_vs_one":
            raise ValueError("one_vs_one multi-class mode does not support "
                             "predicting probability estimates. Use "
                             "one_vs_rest mode instead.")
        X = check_array(X)
        return self.base_estimator_.predict_proba(X)

    @property
    def kernel_(self):
        if self.n_classes_ == 2:
            return self.base_estimator_.kernel_
        else:
            return CompoundKernel(
                [estimator.kernel_
                 for estimator in self.base_estimator_.estimators_])

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.
        In the case of multi-class classification, the mean log-marginal
        likelihood of the one-versus-rest classifiers are returned.
        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or none
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. In the case of multi-class classification, theta may
            be the  hyperparameters of the compound kernel or of an individual
            kernel. In the latter case, all individual kernel get assigned the
            same theta values. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. Note that gradient computation is not supported
            for non-binary classification. If True, theta must not be None.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        check_is_fitted(self, ["classes_", "n_classes_"])

        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        theta = np.asarray(theta)
        if self.n_classes_ == 2:
            return self.base_estimator_.log_marginal_likelihood(
                theta, eval_gradient)
        else:
            if eval_gradient:
                raise NotImplementedError(
                    "Gradient of log-marginal-likelihood not implemented for "
                    "multi-class GPC.")
            estimators = self.base_estimator_.estimators_
            n_dims = estimators[0].kernel_.n_dims
            if theta.shape[0] == n_dims:  # use same theta for all sub-kernels
                return np.mean(
                    [estimator.log_marginal_likelihood(theta)
                     for i, estimator in enumerate(estimators)])
            elif theta.shape[0] == n_dims * self.classes_.shape[0]:
                # theta for compound kernel
                return np.mean(
                    [estimator.log_marginal_likelihood(
                        theta[n_dims * i:n_dims * (i + 1)])
                     for i, estimator in enumerate(estimators)])
            else:
                raise ValueError("Shape of theta must be either %d or %d. "
                                 "Obtained theta with shape %d."
                                 % (n_dims, n_dims * self.classes_.shape[0],
                                    theta.shape[0]))